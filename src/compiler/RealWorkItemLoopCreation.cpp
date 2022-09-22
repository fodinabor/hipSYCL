// LLVM function pass to create loops that run all the work items
// in a work group while respecting barrier synchronization points.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/ValueSymbolTable.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#define DEBUG_TYPE "workitem-loops"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/ParallelRegion.hpp"
#include "hipSYCL/compiler/RealWorkItemLoopCreation.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"

// #define DEBUG_PR_CREATION
// #define DEBUG_WORK_ITEM_LOOPS
// #define DEBUG_REFERENCE_FIXING

#define CONTEXT_ARRAY_ALIGN 64

namespace {
using namespace llvm;
using namespace hipsycl::compiler;

class WorkitemLoops {

public:
  static char ID;

  WorkitemLoops(llvm::Function &F, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT, llvm::LoopInfo &LI,
                VariableUniformityInfo &VUA, SplitterAnnotationInfo &SAA);

private:
  typedef std::vector<llvm::BasicBlock *> BasicBlockVector;
  typedef std::set<llvm::Instruction *> InstructionIndex;
  typedef std::vector<llvm::Instruction *> InstructionVec;
  typedef std::map<std::string, llvm::Instruction *> StrInstructionMap;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;
  SplitterAnnotationInfo &SAA;
  VariableUniformityInfo &VUA;

  ParallelRegion::ParallelRegionVector *OriginalParallelRegions;

  llvm::Type *SizeT;
  StrInstructionMap ContextArrays;

  std::size_t ContextArraysize;

public:
  bool processFunction(llvm::Function &F);

  bool fixUndominatedVariableUses(llvm::Function &F);

private:
  void fixMultiRegionVariables(ParallelRegion *Region);
  void fixMultiRegionAllocas(llvm::Function *F);
  void addContextSaveRestore(llvm::Instruction *I);
  void releaseParallelRegions();
  ParallelRegion::ParallelRegionVector *getParallelRegions(llvm::Function &F);
  void getExitBlocks(llvm::Function &F, llvm::SmallVectorImpl<llvm::BasicBlock *> &ExitBlocks);
  ParallelRegion *createParallelRegionBefore(llvm::BasicBlock *B);

  llvm::Instruction *addContextSave(llvm::Instruction *I, llvm::Instruction *Alloca);
  llvm::Instruction *addContextRestore(llvm::Value *Val, llvm::Instruction *Alloca, llvm::Type *InstType,
                                       bool PoclWrapperStructAdded, llvm::Instruction *Before = NULL,
                                       bool IsAlloca = false);
  llvm::Instruction *getContextArray(llvm::Instruction *Instruction, bool &PoclWrapperStructAdded);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *> createLoopAround(ParallelRegion &region, llvm::BasicBlock *entryBB,
                                                                     llvm::BasicBlock *exitBB, bool peeledFirst,
                                                                     llvm::Value *localIdVar, bool addIncBlock,
                                                                     llvm::Value *DynamicLocalSize);

  ParallelRegion *regionOfBlock(llvm::BasicBlock *BB);
  llvm::BasicBlock *AppendIncBlock(llvm::BasicBlock *after, llvm::PHINode *localIdVar);
  llvm::Value *getLinearWiIndex(llvm::IRBuilder<> &builder, llvm::Module *M, ParallelRegion *region);

  bool shouldNotBeContextSaved(llvm::Instruction *Instr);

  std::map<llvm::Instruction *, unsigned> TempInstructionIds;
  size_t TempInstructionIndex;
  // An alloca in the kernel which stores the first iteration to execute
  // in the inner (dimension 0) loop. This is set to 1 in an peeled iteration
  // to skip the 0, 0, 0 iteration in the loops.
  //   llvm::Value *LocalIdXFirstVar;
  void removeOriginalWILoop();

  llvm::Value *LocalIdXGlobal;
  llvm::Value *LocalIdYGlobal;
  llvm::Value *LocalIdZGlobal;
  llvm::SmallVector<llvm::Value *, 3> LocalSize;
};

WorkitemLoops::WorkitemLoops(llvm::Function &F, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
                             llvm::LoopInfo &LI, VariableUniformityInfo &VUA, SplitterAnnotationInfo &SAA)
    : OriginalParallelRegions(nullptr), DT(DT), PDT(PDT), LI(LI), VUA(VUA), SAA(SAA), TempInstructionIndex(0) {
  llvm::Module *M = F.getParent();
  llvm::DataLayout DL(M);
  SizeT = DL.getLargestLegalIntType(M->getContext());
}

void replacePredecessorsSuccessor(llvm::BasicBlock *Old, llvm::BasicBlock *New) {
  llvm::SmallVector<llvm::BasicBlock *, 4> Preds{llvm::pred_begin(Old), llvm::pred_end(Old)};

  for (auto *Bb : Preds) {
    Bb->getTerminator()->replaceUsesOfWith(Old, New);
  }
}

template <class Predicate>
void addPredecessorsIf(llvm::SmallVectorImpl<llvm::BasicBlock *> &Preds, llvm::BasicBlock *BB, Predicate &&P) {
  for (auto *Pred : llvm::predecessors(BB)) {
    if (P(Pred))
      Preds.push_back(Pred);
  }
}

bool verifyNoBarriers(const llvm::BasicBlock *B, const SplitterAnnotationInfo &SAA) {
  for (auto &I : *B) {
    if (utils::isBarrier(&I, SAA)) {
      HIPSYCL_DEBUG_ERROR << *B << "\n";
      return false;
    }
  }

  return true;
}

void WorkitemLoops::getExitBlocks(llvm::Function &F, llvm::SmallVectorImpl<llvm::BasicBlock *> &ExitBlocks) {
  for (auto &BB : F) {
    auto *T = BB.getTerminator();
    if (T->getNumSuccessors() == 0) {
      // All exits must be barrier blocks.
      if (!utils::blockHasBarrier(&BB, SAA))
        utils::createBarrier(BB.getTerminator(), SAA);
      ExitBlocks.push_back(&BB);
    }
  }
}

void insertLocalIdInit(llvm::BasicBlock *Entry, unsigned X, unsigned Y, unsigned Z) {

  IRBuilder<> Builder(Entry, Entry->getFirstInsertionPt());

  Module *M = Entry->getParent()->getParent();

  //   unsigned long address_bits;
  //   getModuleIntMetadata(*M, "device_address_bits", address_bits);
  auto address_bits = M->getDataLayout().getLargestLegalIntTypeSizeInBits();

  llvm::Type *SizeT = IntegerType::get(M->getContext(), address_bits);

  GlobalVariable *GVX = M->getGlobalVariable(LocalIdGlobalNameX);
  if (GVX != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT, X), GVX);

  GlobalVariable *GVY = M->getGlobalVariable(LocalIdGlobalNameY);
  if (GVY != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT, Y), GVY);

  GlobalVariable *GVZ = M->getGlobalVariable(LocalIdGlobalNameZ);
  if (GVZ != NULL)
    Builder.CreateStore(ConstantInt::get(SizeT, Z), GVZ);
}

ParallelRegion *WorkitemLoops::createParallelRegionBefore(llvm::BasicBlock *B) {
  llvm::SmallVector<llvm::BasicBlock *, 4> PendingBlocks;
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> BlocksInRegion;
  llvm::BasicBlock *RegionEntryBarrier = NULL;
  llvm::BasicBlock *Entry = NULL;
  llvm::BasicBlock *Exit = B->getSinglePredecessor();
  auto NotWILoopEntry = [](llvm::BasicBlock *BB) { return true; };
  addPredecessorsIf(PendingBlocks, B, NotWILoopEntry);

#ifdef DEBUG_PR_CREATION
  llvm::outs().SetUnbuffered();
  HIPSYCL_DEBUG_INFO << "createParallelRegionBefore " << B->getName() << "\n";
#endif

  while (!PendingBlocks.empty()) {
    llvm::BasicBlock *Current = PendingBlocks.back();
    PendingBlocks.pop_back();

#ifdef DEBUG_PR_CREATION
    HIPSYCL_DEBUG_INFO << "considering " << Current->getName() << "\n";
#endif

    // avoid infinite recursion of loops
    if (BlocksInRegion.count(Current) != 0) {
#ifdef DEBUG_PR_CREATION
      HIPSYCL_DEBUG_INFO << "already in the region!\n";
#endif
      continue;
    }

    // If we reach another barrier this must be the
    // parallel region entry.
    if (utils::hasOnlyBarrier(Current, SAA)) {
      if (RegionEntryBarrier == NULL)
        RegionEntryBarrier = Current;
#ifdef DEBUG_PR_CREATION
      HIPSYCL_DEBUG_INFO << "### it's a barrier!\n";
#endif
      continue;
    }

    assert(verifyNoBarriers(Current, SAA) &&
           "Barrier found in a non-barrier block! (forgot barrier canonicalization?)");

#ifdef DEBUG_PR_CREATION
    HIPSYCL_DEBUG_INFO << "added it to the region\n";
#endif
    // Non-barrier block, this must be on the region.
    BlocksInRegion.insert(Current);

    // Add predecessors to pending queue.
    addPredecessorsIf(PendingBlocks, Current, NotWILoopEntry);
  }

  if (BlocksInRegion.empty())
    return NULL;

  // Find the entry node.
  assert(RegionEntryBarrier != NULL);
  for (unsigned Suc = 0, Num = RegionEntryBarrier->getTerminator()->getNumSuccessors(); Suc < Num; ++Suc) {
    llvm::BasicBlock *EntryCandidate = RegionEntryBarrier->getTerminator()->getSuccessor(Suc);
    if (BlocksInRegion.count(EntryCandidate) == 0)
      continue;
    Entry = EntryCandidate;
    break;
  }
  assert(BlocksInRegion.count(Entry) != 0);

  // We got all the blocks in a region, create it.
  return ParallelRegion::Create(BlocksInRegion, Entry, Exit, SAA);
}

/**
 * The main entry to the "parallel region formation", phase which search
 * for the regions between barriers that can be freely parallelized
 * across work-items in the work-group.
 */
ParallelRegion::ParallelRegionVector *WorkitemLoops::getParallelRegions(llvm::Function &F) {
  ParallelRegion::ParallelRegionVector *parallel_regions = new ParallelRegion::ParallelRegionVector;

  SmallVector<BasicBlock *, 4> exit_blocks;
  getExitBlocks(F, exit_blocks);

  // We need to keep track of traversed barriers to detect back edges.
  SmallPtrSet<BasicBlock *, 8> found_barriers;

  // First find all the ParallelRegions in the Function.
  while (!exit_blocks.empty()) {

    // We start on an exit block and process the parallel regions upwards
    // (finding an execution trace).
    BasicBlock *exit = exit_blocks.back();
    exit_blocks.pop_back();

    // already handled
    if (found_barriers.count(exit) != 0)
      continue;

    while (ParallelRegion *PR = createParallelRegionBefore(exit)) {
      assert(PR != NULL && !PR->empty() && "Empty parallel region in kernel (contiguous barriers)!");

      found_barriers.insert(exit);
      exit = NULL;
      parallel_regions->push_back(PR);
      BasicBlock *entry = PR->entryBB();
      int found_predecessors = 0;
      BasicBlock *loop_barrier = NULL;
      for (pred_iterator i = pred_begin(entry), e = pred_end(entry); i != e; ++i) {
        BasicBlock *barrier = (*i);
        if (!found_barriers.count(barrier)) {
          /* If this is a loop header block we might have edges from two
             unprocessed barriers. The one inside the loop (coming from a
             computation block after a branch block) should be processed
             first. */
          std::string bbName = "";
          const bool IS_IN_THE_SAME_LOOP = LI.getLoopFor(barrier) != NULL && LI.getLoopFor(entry) != NULL &&
                                           LI.getLoopFor(entry) == LI.getLoopFor(barrier);

          if (IS_IN_THE_SAME_LOOP) {
#ifdef DEBUG_PR_CREATION
            std::cout << "### found a barrier inside the loop:" << std::endl;
            std::cout << barrier->getName().str() << std::endl;
#endif
            if (loop_barrier != NULL) {
              // there can be multiple latches and each have their barrier,
              // save the previously found inner loop barrier
              exit_blocks.push_back(loop_barrier);
            }
            loop_barrier = barrier;
          } else {
#ifdef DEBUG_PR_CREATION
            std::cout << "### found a barrier:" << std::endl;
            std::cout << barrier->getName().str() << std::endl;
#endif
            exit = barrier;
          }
          ++found_predecessors;
        }
      }

      if (loop_barrier != NULL) {
        /* The secondary barrier to process in case it was a loop
           header. Push it for later processing. */
        if (exit != NULL)
          exit_blocks.push_back(exit);
        /* always process the inner loop regions first */
        if (!found_barriers.count(loop_barrier))
          exit = loop_barrier;
      }

#ifdef DEBUG_PR_CREATION
      std::cout << "### created a ParallelRegion:" << std::endl;
      PR->dumpNames();
      std::cout << std::endl;
#endif

      if (found_predecessors == 0) {
        /* This path has been traversed and we encountered no more
           unprocessed regions. It means we have either traversed all
           paths from the exit or have transformed a loop and thus
           encountered only a barrier that was seen (and thus
           processed) before. */
        break;
      }
      assert((exit != NULL) && "Parallel region without entry barrier!");
    }
  }

#ifdef DEBUG_PR_CREATION
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
#endif
  return parallel_regions;
}

std::pair<llvm::BasicBlock *, llvm::BasicBlock *>
WorkitemLoops::createLoopAround(ParallelRegion &region, llvm::BasicBlock *entryBB, llvm::BasicBlock *exitBB,
                                bool peeledFirst, llvm::Value *localIdVar, bool addIncBlock,
                                llvm::Value *DynamicLocalSize) {
  assert(localIdVar != NULL);

  /*

    Generate a structure like this for each loop level (x,y,z):

    for.init:

    ; if peeledFirst is false:
    store i32 0, i32* %_local_id_x, align 4

    ; if peeledFirst is true (assume the 0,0,0 iteration has been executed earlier)
    ; assume _local_id_x_first is is initialized to 1 in the peeled pregion copy
    store _local_id_x_first, i32* %_local_id_x, align 4
    store i32 0, %_local_id_x_first

    br label %for.body

    for.body:

    ; the parallel region code here

    br label %for.inc

    for.inc:

    ; Separated inc and cond check blocks for easier loop unrolling later on.
    ; Can then chain N times for.body+for.inc to unroll.

    %2 = load i32* %_local_id_x, align 4
    %inc = add nsw i32 %2, 1

    store i32 %inc, i32* %_local_id_x, align 4
    br label %for.cond

    for.cond:

    ; loop header, compare the id to the local size
    %0 = load i32* %_local_id_x, align 4
    %cmp = icmp ult i32 %0, i32 123
    br i1 %cmp, label %for.body, label %for.end

    for.end:

    OPTIMIZE: Use a separate iteration variable across all the loops to iterate the context
    data arrays to avoid needing multiplications to find the correct location, and to
    enable easy vectorization of loading the context data when there are parallel iterations.
  */

  llvm::BasicBlock *loopBodyEntryBB = entryBB;
  llvm::LLVMContext &C = loopBodyEntryBB->getContext();
  llvm::Function *F = loopBodyEntryBB->getParent();
  loopBodyEntryBB->setName(std::string("pregion_for_entry.") + entryBB->getName().str());

  assert(exitBB->getTerminator()->getNumSuccessors() == 1);

  llvm::BasicBlock *oldExit = exitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock *forInitBB = BasicBlock::Create(C, "pregion_for_init", F, loopBodyEntryBB);

  llvm::BasicBlock *loopEndBB = BasicBlock::Create(C, "pregion_for_end", F, exitBB);

  llvm::BasicBlock *forCondBB = BasicBlock::Create(C, "pregion_for_cond", F, exitBB);

  DT.reset();
  DT.recalculate(*F);

  /* Collect the basic blocks in the parallel region that dominate the
     exit. These are used in determining whether load instructions may
     be executed unconditionally in the parallel loop (see below). */
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> dominatesExitBB;
  for (auto bb : region) {
    if (DT.dominates(bb, exitBB)) {
      dominatesExitBB.insert(bb);
    }
  }

  //  F->viewCFG();
  /* Fix the old edges jumping to the region to jump to the basic block
     that starts the created loop. Back edges should still point to the
     old basic block so we preserve the old loops. */
  BasicBlockVector preds;
  llvm::pred_iterator PI = llvm::pred_begin(entryBB), E = llvm::pred_end(entryBB);

  for (; PI != E; ++PI) {
    llvm::BasicBlock *bb = *PI;
    preds.push_back(bb);
  }

  for (BasicBlockVector::iterator i = preds.begin(); i != preds.end(); ++i) {
    llvm::BasicBlock *bb = *i;
    /* Do not fix loop edges inside the region. The loop
       is replicated as a whole to the body of the wi-loop.*/
    if (DT.dominates(loopBodyEntryBB, bb))
      continue;
    bb->getTerminator()->replaceUsesOfWith(loopBodyEntryBB, forInitBB);
  }

  IRBuilder<> builder(loopBodyEntryBB, loopBodyEntryBB->getFirstInsertionPt());

  auto IndVar = builder.CreatePHI(SizeT, 2, "IndVar");

  builder.SetInsertPoint(forInitBB);
  if (peeledFirst) {
    llvm::Value *cmpResult;
    cmpResult = builder.CreateICmpULT(ConstantInt::get(SizeT, 1), DynamicLocalSize);
    IndVar->addIncoming(ConstantInt::get(SizeT, 1), forInitBB);

    builder.CreateCondBr(cmpResult, loopBodyEntryBB, loopEndBB);
  } else {
    IndVar->addIncoming(ConstantInt::get(SizeT, 0), forInitBB);
    builder.CreateBr(loopBodyEntryBB);
  }

  HIPSYCL_DEBUG_INFO << *exitBB << *oldExit << *forCondBB << "\n";
  exitBB->getTerminator()->replaceUsesOfWith(oldExit, forCondBB);

  //   if (addIncBlock) {
  // AppendIncBlock(exitBB, IndVar);
  //   }
  builder.SetInsertPoint(forCondBB);

  auto Inced = builder.CreateAdd(IndVar, ConstantInt::get(SizeT, 1));
  IndVar->addIncoming(Inced, forCondBB);

  llvm::Value *cmpResult = builder.CreateICmpULT(Inced, DynamicLocalSize);

  Instruction *loopBranch = builder.CreateCondBr(cmpResult, loopBodyEntryBB, loopEndBB);

  /* Add the metadata to mark a parallel loop. The metadata
     refer to a loop-unique dummy metadata that is not merged
     automatically. */

  /* This creation of the identifier metadata is copied from
     LLVM's MDBuilder::createAnonymousTBAARoot(). */

  MDNode *Dummy = MDNode::getTemporary(C, ArrayRef<Metadata *>()).release();
#ifdef LLVM_OLDER_THAN_8_0
  MDNode *Root = MDNode::get(C, Dummy);
#else
  MDNode *AccessGroupMD = MDNode::getDistinct(C, {});
  MDNode *ParallelAccessMD = MDNode::get(C, {MDString::get(C, "llvm.loop.parallel_accesses"), AccessGroupMD});
  auto *MDWorkItemLoop = llvm::MDNode::get(C, {llvm::MDString::get(C, MDKind::WorkItemLoop)});
  MDNode *Root = MDNode::get(C, {Dummy, ParallelAccessMD, MDWorkItemLoop});

#endif

  // At this point we have
  //   !0 = metadata !{}            <- dummy
  //   !1 = metadata !{metadata !0} <- root
  // Replace the dummy operand with the root node itself and delete the dummy.
  Root->replaceOperandWith(0, Root);
  MDNode::deleteTemporary(Dummy);
  // We now have
  //   !1 = metadata !{metadata !1} <- self-referential root
  loopBranch->setMetadata("llvm.loop", Root);

  auto IsLoadUnconditionallySafe = [&dominatesExitBB](llvm::Instruction *insn) -> bool {
    assert(insn->mayReadFromMemory());
    // Checks that the instruction isn't in a conditional block.
    return dominatesExitBB.count(insn->getParent());
  };

#ifdef LLVM_OLDER_THAN_8_0
  region.AddParallelLoopMetadata(Root, IsLoadUnconditionallySafe);
#else
  region.AddParallelLoopMetadata(AccessGroupMD, IsLoadUnconditionallySafe);
#endif

  builder.SetInsertPoint(loopEndBB);
  builder.CreateBr(oldExit);

  llvm::ValueToValueMapTy VMap;
  VMap[localIdVar] = IndVar;
  llvm::SmallVector<llvm::BasicBlock *> Blocks{region.begin(), region.end()};
  llvm::remapInstructionsInBlocks(Blocks, VMap);

  return std::make_pair(forInitBB, loopEndBB);
}

ParallelRegion *WorkitemLoops::regionOfBlock(llvm::BasicBlock *bb) {
  for (ParallelRegion::ParallelRegionVector::iterator i = OriginalParallelRegions->begin(),
                                                      e = OriginalParallelRegions->end();
       i != e; ++i) {
    ParallelRegion *region = (*i);
    if (region->HasBlock(bb))
      return region;
  }
  return NULL;
}

void WorkitemLoops::releaseParallelRegions() {
  if (OriginalParallelRegions) {
    for (auto i = OriginalParallelRegions->begin(), e = OriginalParallelRegions->end(); i != e; ++i) {
      ParallelRegion *p = *i;
      delete p;
    }

    delete OriginalParallelRegions;
    OriginalParallelRegions = nullptr;
  }
}

void purgeLifetime(ParallelRegion &region) {
  llvm::SmallVector<llvm::Instruction *, 8> ToDelete;
  for (auto *BB : region)
    for (auto &I : *BB)
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (CI->getCalledFunction())
          if (CI->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::lifetime_start ||
              CI->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::lifetime_end)
            ToDelete.push_back(CI);

  for (auto *I : ToDelete)
    I->eraseFromParent();

  //  // remove dead bitcasts
  //  for (auto *BB : Cfg.getNewBlocks())
  //    llvm::SimplifyInstructionsInBlock(BB);
}

bool WorkitemLoops::processFunction(Function &F) {
  //   Kernel *K = cast<Kernel>(&F);

  llvm::Module *M = F.getParent();

  releaseParallelRegions();

  OriginalParallelRegions = getParallelRegions(F);

  auto Dim = utils::getRangeDim(F);
  LocalSize = utils::getLocalSizeValues(F, Dim);
  {
    llvm::SmallVector<llvm::BasicBlock *> Blocks;
    Blocks.reserve(std::distance(F.begin(), F.end()));
    std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });
    // utils::moveAllocasToEntry(F, Blocks);
    utils::moveGlobalVarLoadsToEntry(F, Blocks, LocalIdGlobalNameX);
    utils::moveGlobalVarLoadsToEntry(F, Blocks, LocalIdGlobalNameY);
    utils::moveGlobalVarLoadsToEntry(F, Blocks, LocalIdGlobalNameZ);
    LocalIdXGlobal = M->getNamedGlobal(LocalIdGlobalNameX);
    LocalIdYGlobal = M->getNamedGlobal(LocalIdGlobalNameY);
    LocalIdZGlobal = M->getNamedGlobal(LocalIdGlobalNameZ);
  }

#ifdef DUMP_CFGS
  F.dump();
  dumpCFG(F, F.getName().str() + "_before_wiloops.dot", OriginalParallelRegions);
#endif

  IRBuilder<> builder(&*(F.getEntryBlock().getFirstInsertionPt()));
  //   LocalIdXFirstVar = builder.CreateAlloca(SizeT, 0, ".pocl.local_id_x_init");

  //  F.viewCFGOnly();

#if 0
  std::cerr << "### Original" << std::endl;
  F.viewCFGOnly();
#endif

#if 0
  for (ParallelRegion::ParallelRegionVector::iterator
           i = OriginalParallelRegions->begin(),
           e = OriginalParallelRegions->end();
       i != e; ++i) 
  {
    ParallelRegion *region = (*i);
    region->InjectRegionPrintF();
    region->InjectVariablePrintouts();
  }
#endif

  /* Count how many parallel regions share each entry node to
     detect diverging regions that need to be peeled. */
  std::map<llvm::BasicBlock *, int> entryCounts;

  for (ParallelRegion::ParallelRegionVector::iterator i = OriginalParallelRegions->begin(),
                                                      e = OriginalParallelRegions->end();
       i != e; ++i) {
    ParallelRegion *region = (*i);
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### Adding context save/restore for PR: ";
    region->dumpNames();
#endif
    purgeLifetime(*region);

    fixMultiRegionVariables(region);
    entryCounts[region->entryBB()]++;
  }

#if 0
  std::cerr << "### After context code addition:" << std::endl;
  F.viewCFG();
#endif
  std::map<ParallelRegion *, bool> peeledRegion;
  for (ParallelRegion::ParallelRegionVector::iterator i = OriginalParallelRegions->begin(),
                                                      e = OriginalParallelRegions->end();
       i != e; ++i) {

    llvm::ValueToValueMapTy reference_map;
    ParallelRegion *original = (*i);

#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### handling region:" << std::endl;
    original->dumpNames();
    // F.viewCFGOnly();
#endif

    /* In case of conditional barriers, the first iteration
       has to be peeled so we know which branch to execute
       with the work item loop. In case there are more than one
       parallel region sharing an entry BB, it's a diverging
       region.

       Post dominance of entry by exit does not work in case the
       region is inside a loop and the exit block is in the path
       towards the loop exit (and the function exit).
    */
    bool peelFirst = entryCounts[original->entryBB()] > 1;

    peeledRegion[original] = peelFirst;

    std::pair<llvm::BasicBlock *, llvm::BasicBlock *> l;
    // the original predecessor nodes of which successor
    // should be fixed if not peeling
    BasicBlockVector preds;

    bool unrolled = false;
    if (peelFirst) {
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### conditional region, peeling the first iteration" << std::endl;
#endif
      ParallelRegion *replica = original->replicate(reference_map, ".peeled_wi");
      replica->chainAfter(original);
      replica->purge();

      original = replica;

      l = std::make_pair(replica->entryBB(), replica->exitBB());
    } else {
      llvm::pred_iterator PI = llvm::pred_begin(original->entryBB()), E = llvm::pred_end(original->entryBB());

      for (; PI != E; ++PI) {
        llvm::BasicBlock *bb = *PI;
        if (DT.dominates(original->entryBB(), bb) && (regionOfBlock(original->entryBB()) == regionOfBlock(bb)))
          continue;
        preds.push_back(bb);
      }

      /* Find a two's exponent unroll count, if available. */
      l = std::make_pair(original->entryBB(), original->exitBB());
    }

    llvm::SmallVector<llvm::BasicBlock *> Blocks{original->begin(), original->end()};

    if (Dim > 2) {
      l = createLoopAround(*original, l.first, l.second, Dim == 3 && peelFirst,
                           utils::getLoadForGlobalVariable(F, LocalIdGlobalNameZ), !unrolled, LocalSize[2]);
    }

    if (Dim > 1) {
      l = createLoopAround(*original, l.first, l.second, Dim == 2 && peelFirst,
                           utils::getLoadForGlobalVariable(F, LocalIdGlobalNameY), !unrolled, LocalSize[1]);
    }

    l = createLoopAround(*original, l.first, l.second, Dim == 1 && peelFirst,
                         utils::getLoadForGlobalVariable(F, LocalIdGlobalNameX), !unrolled, LocalSize[0]);

    /* Loop edges coming from another region mean B-loops which means
       we have to fix the loop edge to jump to the beginning of the wi-loop
       structure, not its body. This has to be done only for non-peeled
       blocks as the semantics is correct in the other case (the jump is
       to the beginning of the peeled iteration). */
    if (!peelFirst) {
      for (BasicBlockVector::iterator i = preds.begin(); i != preds.end(); ++i) {
        llvm::BasicBlock *bb = *i;
        bb->getTerminator()->replaceUsesOfWith(original->entryBB(), l.first);
      }
    }
    HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  }

  // for the peeled regions we need to add a prologue
  // that initializes the local ids and the first iteration
  // counter
  for (ParallelRegion::ParallelRegionVector::iterator i = OriginalParallelRegions->begin(),
                                                      e = OriginalParallelRegions->end();
       i != e; ++i) {
    ParallelRegion *pr = (*i);

    if (!peeledRegion[pr])
      continue;
    llvm::ValueToValueMapTy VMap;
    for (int I = 0; I < Dim; ++I)
      VMap[utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[I])] = ConstantInt::get(SizeT, 0);

    llvm::SmallVector<llvm::BasicBlock *> Blocks{pr->begin(), pr->end()};
    llvm::remapInstructionsInBlocks(Blocks, VMap);

    // insertLocalIdInit(pr->entryBB(), 0, 0, 0);
    // builder.SetInsertPoint(&*(pr->entryBB()->getFirstInsertionPt()));
    // builder.CreateStore(ConstantInt::get(SizeT, 1), LocalIdXFirstVar);
  }

  //   insertLocalIdInit(&F.getEntryBlock(), 0, 0, 0);

#if 0
  F.viewCFG();
#endif

  return true;
}

/*
 * Add context save/restore code to variables that are defined in
 * the given region and are used outside the region.
 *
 * Each such variable gets a slot in the stack frame. The variable
 * is restored from the stack whenever it's used.
 *
 */
void WorkitemLoops::fixMultiRegionVariables(ParallelRegion *region) {
  InstructionIndex instructionsInRegion;
  InstructionVec instructionsToFix;

  /* Construct an index of the region's instructions so it's
     fast to figure out if the variable uses are all
     in the region. */
  for (BasicBlockVector::iterator i = region->begin(); i != region->end(); ++i) {
    llvm::BasicBlock *bb = *i;
    for (llvm::BasicBlock::iterator instr = bb->begin(); instr != bb->end(); ++instr) {
      llvm::Instruction *instruction = &*instr;
      instructionsInRegion.insert(instruction);
    }
  }

  /* Find all the instructions that define new values and
     check if they need to be context saved. */
  for (BasicBlockVector::iterator i = region->begin(); i != region->end(); ++i) {
    llvm::BasicBlock *bb = *i;
    for (llvm::BasicBlock::iterator instr = bb->begin(); instr != bb->end(); ++instr) {
      llvm::Instruction *instruction = &*instr;

      if (shouldNotBeContextSaved(&*instr))
        continue;

      for (Instruction::use_iterator ui = instruction->use_begin(), ue = instruction->use_end(); ui != ue; ++ui) {
        llvm::Instruction *user = dyn_cast<Instruction>(ui->getUser());

        if (user == NULL)
          continue;
        // If the instruction is used outside this region inside another
        // region (not in a regionless BB like the B-loop construct BBs),
        // need to context save it.
        // Allocas (private arrays) should be privatized always. Otherwise
        // we end up reading the same array, but replicating the GEP to that.
        if (isa<AllocaInst>(instruction) || (instructionsInRegion.find(user) == instructionsInRegion.end() &&
                                             regionOfBlock(user->getParent()) != NULL)) {
          instructionsToFix.push_back(instruction);
          break;
        }
      }
    }
  }

  /* Finally, fix the instructions. */
  for (auto *I : instructionsToFix) {
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### adding context/save restore for" << std::endl;
    HIPSYCL_DEBUG_INFO << *I;
#endif
    addContextSaveRestore(I);
  }
}

llvm::Value *WorkitemLoops::getLinearWiIndex(llvm::IRBuilder<> &builder, llvm::Module *M, ParallelRegion *region) {
  auto *F = region->entryBB()->getParent();

  if (auto Idx = region->GetContiguousIdx())
    return Idx;

  llvm::IRBuilder Builder{region->entryBB()->getFirstNonPHI()};

  auto Idx = utils::getLoadForGlobalVariable(*F, LocalIdGlobalNames[0]);
  for (size_t D = 1; D < LocalSize.size(); ++D) {
    const std::string Suffix = (llvm::Twine{DimName[D]}).str();

    Idx = Builder.CreateMul(Idx, LocalSize[D], "idx.mul." + Suffix, true);
    Idx = Builder.CreateAdd(utils::getLoadForGlobalVariable(*F, LocalIdGlobalNames[D]), Idx, "idx.add." + Suffix, true);
  }
  region->SetContiguousIdx(Idx);
  return Idx;
}

llvm::Instruction *WorkitemLoops::addContextSave(llvm::Instruction *instruction, llvm::Instruction *alloca) {

  if (isa<AllocaInst>(instruction)) {
    /* If the variable to be context saved is itself an alloca,
       we have created one big alloca that stores the data of all the
       work-items and return pointers to that array. Thus, we need
       no initialization code other than the context data alloca itself. */
    return NULL;
  }

  /* Save the produced variable to the array. */
  BasicBlock::iterator definition = (dyn_cast<Instruction>(instruction))->getIterator();
  ++definition;
  while (isa<PHINode>(definition))
    ++definition;

  IRBuilder<> builder(&*definition);
  std::vector<llvm::Value *> gepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *region = regionOfBlock(instruction->getParent());
  assert("Adding context save outside any region produces illegal code." && region != NULL);

  Module *M = alloca->getParent()->getParent()->getParent();
  gepArgs.push_back(getLinearWiIndex(builder, M, region));

  return builder.CreateStore(instruction,
                             builder.CreateGEP(alloca->getType()->getPointerElementType(), alloca, gepArgs));
}

llvm::Instruction *WorkitemLoops::addContextRestore(llvm::Value *val, llvm::Instruction *alloca, llvm::Type *InstType,
                                                    bool PoclWrapperStructAdded, llvm::Instruction *before,
                                                    bool isAlloca) {
  assert(val != NULL);
  assert(alloca != NULL);
  IRBuilder<> builder(alloca);
  if (before != NULL) {
    builder.SetInsertPoint(before);
  } else if (isa<Instruction>(val)) {
    builder.SetInsertPoint(dyn_cast<Instruction>(val));
    before = dyn_cast<Instruction>(val);
  } else {
    assert(false && "Unknown context restore location!");
  }

  std::vector<llvm::Value *> gepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *region = regionOfBlock(before->getParent());
  assert("Adding context save outside any region produces illegal code." && region != NULL);

  Module *M = alloca->getParent()->getParent()->getParent();
  gepArgs.push_back(getLinearWiIndex(builder, M, region));

  if (PoclWrapperStructAdded)
    gepArgs.push_back(ConstantInt::get(Type::getInt32Ty(alloca->getContext()), 0));

  llvm::Instruction *gep =
      dyn_cast<Instruction>(builder.CreateGEP(alloca->getType()->getPointerElementType(), alloca, gepArgs));
  if (isAlloca) {
    /* In case the context saved instruction was an alloca, we created a
       context array with pointed-to elements, and now want to return a
       pointer to the elements to emulate the original alloca. */
    return gep;
  }
  return builder.CreateLoad(InstType, gep);
}

/**
 * Returns the context array (alloca) for the given Value, creates it if not
 * found.
 */
llvm::Instruction *WorkitemLoops::getContextArray(llvm::Instruction *instruction, bool &PoclWrapperStructAdded) {
  PoclWrapperStructAdded = false;
  /*
   * Unnamed temp instructions need a generated name for the
   * context array. Create one using a running integer.
   */
  std::ostringstream var;
  var << ".";

  if (std::string(instruction->getName().str()) != "") {
    var << instruction->getName().str();
  } else if (TempInstructionIds.find(instruction) != TempInstructionIds.end()) {
    var << TempInstructionIds[instruction];
  } else {
    TempInstructionIds[instruction] = TempInstructionIndex++;
    var << TempInstructionIds[instruction];
  }

  var << ".pocl_context";
  std::string varName = var.str();

  if (ContextArrays.find(varName) != ContextArrays.end())
    return ContextArrays[varName];

  BasicBlock &bb = instruction->getParent()->getParent()->getEntryBlock();
  IRBuilder<> builder(&*(bb.getFirstInsertionPt()));
  Function *FF = instruction->getParent()->getParent();
  Module *M = instruction->getParent()->getParent()->getParent();
  LLVMContext &C = M->getContext();
  const llvm::DataLayout &Layout = M->getDataLayout();
  DICompileUnit *CU = nullptr;
  std::unique_ptr<DIBuilder> DB;
#ifndef LLVM_OLDER_THAN_7_0
  if (M->debug_compile_units_begin() != M->debug_compile_units_end()) {
    CU = *M->debug_compile_units_begin();
    DB = std::unique_ptr<DIBuilder>{new DIBuilder(*M, true, CU)};
  }
#endif

  // find the debug metadata corresponding to this variable
  Value *DebugVal = nullptr;
  IntrinsicInst *DebugCall = nullptr;
#ifndef LLVM_OLDER_THAN_7_0
  if (CU) {
    for (BasicBlock &BB : (*FF)) {
      for (Instruction &I : BB) {
        IntrinsicInst *CI = dyn_cast<IntrinsicInst>(&I);
        if (CI && (CI->getIntrinsicID() == llvm::Intrinsic::dbg_declare)) {
          Metadata *Meta = cast<MetadataAsValue>(CI->getOperand(0))->getMetadata();
          if (isa<ValueAsMetadata>(Meta)) {
            Value *V = cast<ValueAsMetadata>(Meta)->getValue();
            if (instruction == V) {
              DebugVal = V;
              DebugCall = CI;
              break;
            }
          }
        }
      }
    }
  }
#endif

#ifdef DEBUG_WORK_ITEM_LOOPS
  if (DebugVal && DebugCall) {
    llvm::errs() << "### DI INTRIN: \n";
    llvm::errs() << *DebugCall;
    llvm::errs() << "### DI VALUE:  \n";
    llvm::errs() << *DebugVal;
  }
#endif

  llvm::Type *elementType;
  if (isa<AllocaInst>(instruction)) {
    /* If the variable to be context saved was itself an alloca,
       create one big alloca that stores the data of all the
       work-items and directly return pointers to that array.
       This enables moving all the allocas to the entry node without
       breaking the parallel loop.
       Otherwise we would rely on a dynamic alloca to allocate
       unique stack space to all the work-items when its wiloop
       iteration is executed. */
    elementType = dyn_cast<AllocaInst>(instruction)->getType()->getElementType();
  } else {
    elementType = instruction->getType();
  }

  /* 3D context array. In case the elementType itself is an array or struct,
   * we must take into account it could be alloca-ed with alignment and loads
   * or stores might use vectorized instructions expecting proper alignment.
   * Because of that, we cannot simply allocate x*y*z*(size), we must
   * enlarge the type to fit the alignment. */
  Type *AllocType = elementType;
  AllocaInst *InstCast = dyn_cast<AllocaInst>(instruction);
  if (InstCast) {
    unsigned Alignment = InstCast->getAlignment();

    uint64_t StoreSize = Layout.getTypeStoreSize(InstCast->getAllocatedType());

    if ((Alignment > 1) && (StoreSize & (Alignment - 1))) {
      uint64_t AlignedSize = (StoreSize & (~(Alignment - 1))) + Alignment;
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### unaligned type found: aligning " << StoreSize << " to " << AlignedSize << "\n";
#endif
      assert(AlignedSize > StoreSize);
      uint64_t RequiredExtraBytes = AlignedSize - StoreSize;

      if (isa<ArrayType>(elementType)) {

        ArrayType *StructPadding = ArrayType::get(Type::getInt8Ty(M->getContext()), RequiredExtraBytes);

        std::vector<Type *> PaddedStructElements;
        PaddedStructElements.push_back(elementType);
        PaddedStructElements.push_back(StructPadding);
        const ArrayRef<Type *> NewStructElements(PaddedStructElements);
        AllocType = StructType::get(M->getContext(), NewStructElements, true);
        PoclWrapperStructAdded = true;
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);

      } else if (isa<StructType>(elementType)) {
        StructType *OldStruct = dyn_cast<StructType>(elementType);

        ArrayType *StructPadding = ArrayType::get(Type::getInt8Ty(M->getContext()), RequiredExtraBytes);
        std::vector<Type *> PaddedStructElements;
        for (unsigned j = 0; j < OldStruct->getNumElements(); j++)
          PaddedStructElements.push_back(OldStruct->getElementType(j));
        PaddedStructElements.push_back(StructPadding);
        const ArrayRef<Type *> NewStructElements(PaddedStructElements);
        AllocType = StructType::get(OldStruct->getContext(), NewStructElements, OldStruct->isPacked());
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);
      }
    }
  }

  llvm::AllocaInst *Alloca = nullptr;
  // Value *NumberOfWorkItems = LocalSize[0];
  // for (int I = 1; I < LocalSize.size(); ++I)
  //   NumberOfWorkItems = builder.CreateBinOp(Instruction::Mul, NumberOfWorkItems, LocalSize[I], "num_wi");

  Alloca = builder.CreateAlloca(AllocType, llvm::ConstantInt::get(SizeT, 1024), varName);

  /* Align the context arrays to stack to enable wide vectors
     accesses to them. Also, LLVM 3.3 seems to produce illegal
     code at least with Core i5 when aligned only at the element
     size. */
  Alloca->setAlignment(
#ifndef LLVM_OLDER_THAN_10_0
#ifndef LLVM_OLDER_THAN_11_0
      llvm::Align(
#else
      llvm::MaybeAlign(
#endif
#endif
          CONTEXT_ARRAY_ALIGN
#ifndef LLVM_OLDER_THAN_10_0
          )
#endif
  );

  ContextArrays[varName] = Alloca;
  return Alloca;
}

/**
 * Adds context save/restore code for the value produced by the
 * given instruction.
 *
 * TODO: add only one restore per variable per region.
 * TODO: add only one load of the id variables per region.
 * Could be done by having a context restore BB in the beginning of the
 * region and a context save BB at the end.
 * TODO: ignore work group variables completely (the iteration variables)
 * The LLVM should optimize these away but it would improve
 * the readability of the output during debugging.
 * TODO: rematerialize some values such as extended values of global
 * variables (especially global id which is computed from local id) or kernel
 * argument values instead of allocating stack space for them
 */
void WorkitemLoops::addContextSaveRestore(llvm::Instruction *instruction) {

  /* Allocate the context data array for the variable. */
  bool PoclWrapperStructAdded = false;
  llvm::Instruction *alloca = getContextArray(instruction, PoclWrapperStructAdded);
  llvm::Instruction *theStore = addContextSave(instruction, alloca);

  InstructionVec uses;
  /* Restore the produced variable before each use to ensure the correct context
     copy is used.

     We could add the restore only to other regions outside the
     variable defining region and use the original variable in the defining
     region due to the SSA virtual registers being unique. However,
     alloca variables can be redefined also in the same region, thus we
     need to ensure the correct alloca context position is written, not
     the original unreplicated one. These variables can be generated by
     volatile variables, private arrays, and due to the PHIs to allocas
     pass.
  */

  /* Find out the uses to fix first as fixing them invalidates
     the iterator. */
  for (Instruction::use_iterator ui = instruction->use_begin(), ue = instruction->use_end(); ui != ue; ++ui) {
    llvm::Instruction *user = cast<Instruction>(ui->getUser());
    if (user == NULL)
      continue;
    if (user == theStore)
      continue;
    uses.push_back(user);
  }

  for (InstructionVec::iterator i = uses.begin(); i != uses.end(); ++i) {
    Instruction *user = *i;
    Instruction *contextRestoreLocation = user;
    /* If the user is in a block that doesn't belong to a region,
       the variable itself must be a "work group variable", that is,
       not dependent on the work item. Most likely an iteration
       variable of a for loop with a barrier. */
    if (regionOfBlock(user->getParent()) == NULL)
      continue;

    PHINode *phi = dyn_cast<PHINode>(user);
    if (phi != NULL) {
      /* In case of PHI nodes, we cannot just insert the context
         restore code before it in the same basic block because it is
         assumed there are no non-phi Instructions before PHIs which
         the context restore code constitutes to. Add the context
         restore to the incomingBB instead.

         There can be values in the PHINode that are incoming
         from another region even though the decision BB is within the region.
         For those values we need to add the context restore code in the
         incoming BB (which is known to be inside the region due to the
         assumption of not having to touch PHI nodes in PRentry BBs).
      */

      /* PHINodes at region entries are broken down earlier. */
      assert("Cannot add context restore for a PHI node at the region entry!" &&
             regionOfBlock(phi->getParent())->entryBB() != phi->getParent());
#ifdef DEBUG_WORK_ITEM_LOOPS
      HIPSYCL_DEBUG_INFO << "### adding context restore code before PHI\n" << *user << "\n";
      HIPSYCL_DEBUG_INFO << "### in BB:\n" << *user->getParent() << "\n";
#endif
      BasicBlock *incomingBB = NULL;
      for (unsigned incoming = 0; incoming < phi->getNumIncomingValues(); ++incoming) {
        Value *val = phi->getIncomingValue(incoming);
        BasicBlock *bb = phi->getIncomingBlock(incoming);
        if (val == instruction)
          incomingBB = bb;
      }
      assert(incomingBB != NULL);
      contextRestoreLocation = incomingBB->getTerminator();
    }
    llvm::Value *loadedValue = addContextRestore(user, alloca, instruction->getType(), PoclWrapperStructAdded,
                                                 contextRestoreLocation, isa<AllocaInst>(instruction));
    user->replaceUsesOfWith(instruction, loadedValue);

#ifdef DEBUG_WORK_ITEM_LOOPS
    HIPSYCL_DEBUG_INFO << "### done, the user was converted to:\n" << *user << "\n";
#endif
  }
}

bool WorkitemLoops::shouldNotBeContextSaved(llvm::Instruction *instr) {
#ifdef DEBUG_WORK_ITEM_LOOPS
  HIPSYCL_DEBUG_INFO << "### should context save " << *instr << "?\n";
#endif
  /*
    _local_id loads should not be replicated as it leads to
    problems in conditional branch case where the header node
    of the region is shared across the branches and thus the
    header node's ID loads might get context saved which leads
    to egg-chicken problems.
  */
  if (isa<BranchInst>(instr))
    return true;

  llvm::LoadInst *load = dyn_cast<llvm::LoadInst>(instr);
  if (load != NULL && (load->getPointerOperand() == LocalIdZGlobal || load->getPointerOperand() == LocalIdYGlobal ||
                       load->getPointerOperand() == LocalIdXGlobal))
    return true;

  /* In case of uniform variables (same for all work-items),
     there is no point to create a context array slot for them,
     but just use the original value everywhere.

     Allocas are problematic: they include the de-phi induction
     variables of the b-loops. In those case each work item
     has a separate loop iteration variable in the LLVM IR but
     which is really a parallel region loop invariant. But
     because we cannot separate such loop invariant variables
     at this point sensibly, let's just replicate the iteration
     variable to each work item and hope the latter optimizations
     reduce them back to a single induction variable outside the
     parallel loop.
  */
  if (!VUA.shouldBePrivatized(instr->getParent()->getParent(), instr)) {
#ifdef DEBUG_WORK_ITEM_LOOPS
    HIPSYCL_DEBUG_INFO << "### based on VUA, not context saving: " << *instr << "\n";
#endif
    return true;
  }
#ifdef DEBUG_WORK_ITEM_LOOPS
  HIPSYCL_DEBUG_INFO << "### indeed context saving: " << *instr << "\n";
#endif
  return false;
}

llvm::BasicBlock *WorkitemLoops::AppendIncBlock(llvm::BasicBlock *after, llvm::PHINode *localId) {
  llvm::LLVMContext &C = after->getContext();

  llvm::BasicBlock *oldExit = after->getTerminator()->getSuccessor(0);
  assert(oldExit != NULL);

  llvm::BasicBlock *forIncBB = BasicBlock::Create(C, "pregion_for_inc", after->getParent());

  after->getTerminator()->replaceUsesOfWith(oldExit, forIncBB);

  IRBuilder<> builder(oldExit);

  builder.SetInsertPoint(forIncBB);
  /* Create the iteration variable increment */
  auto Inced = builder.CreateAdd(localId, ConstantInt::get(SizeT, 1));
  localId->addIncoming(Inced, oldExit);

  builder.CreateBr(oldExit);

  return forIncBB;
}

bool dominatesUse(llvm::DominatorTree &DT, llvm::Instruction &I, unsigned Idx) {
  llvm::Instruction *Op = llvm::cast<llvm::Instruction>(I.getOperand(Idx));
  llvm::BasicBlock *OpBlock = Op->getParent();
  llvm::PHINode *PN = llvm::dyn_cast<llvm::PHINode>(&I);

  // DT can handle non phi instructions for us.
  if (!PN) {
    // Definition must dominate use unless use is unreachable!
    return Op->getParent() == I.getParent() || DT.dominates(Op, &I);
  }

  // PHI nodes are more difficult than other nodes because they actually
  // "use" the value in the predecessor basic blocks they correspond to.
  unsigned J = llvm::PHINode::getIncomingValueNumForOperand(Idx);
  llvm::BasicBlock *PredBB = PN->getIncomingBlock(J);
  return (PredBB && DT.dominates(OpBlock, PredBB));
}

/* Fixes the undominated variable uses.

   These appear when a conditional barrier kernel is replicated to
   form a copy of the *same basic block* in the alternative
   "barrier path".

   E.g., from

   A -> [exit], A -> B -> [exit]

   a replicated CFG as follows, is created:

   A1 -> (T) A2 -> [exit1],  A1 -> (F) A2' -> B1, B2 -> [exit2]

   The regions are correct because of the barrier semantics
   of "all or none". In case any barrier enters the [exit1]
   from A1, all must (because there's a barrier in the else
   branch).

   Here at A2 and A2' one creates the same variables.
   However, B2 does not know which copy
   to refer to, the ones created in A2 or ones in A2' (correct).
   The mapping data contains only one possibility, the
   one that was placed there last. Thus, the instructions in B2
   might end up referring to the variables defined in A2
   which do not nominate them.

   The variable references are fixed by exploiting the knowledge
   of the naming convention of the cloned variables.

   One potential alternative way would be to collect the refmaps per BB,
   not globally. Then as a final phase traverse through the
   basic blocks starting from the beginning and propagating the
   reference data downwards, the data from the new BB overwriting
   the old one. This should ensure the reachability without
   the costly dominance analysis.
*/
bool WorkitemLoops::fixUndominatedVariableUses(llvm::Function &F) {
  bool Changed = false;

  DT.reset();
  DT.recalculate(F);

  for (llvm::Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    llvm::BasicBlock *Bb = &*I;
    for (auto &Ins : *Bb) {
      for (unsigned Opr = 0; Opr < Ins.getNumOperands(); ++Opr) {
        if (!llvm::isa<llvm::Instruction>(Ins.getOperand(Opr)))
          continue;
        llvm::Instruction *Operand = llvm::cast<llvm::Instruction>(Ins.getOperand(Opr));
        if (dominatesUse(DT, Ins, Opr))
          continue;
#ifdef DEBUG_REFERENCE_FIXING
        HIPSYCL_DEBUG_INFO << "### dominance error!\n";
        HIPSYCL_DEBUG_EXECUTE_INFO(Operand->print(llvm::outs()); llvm::outs() << "\n";)
        HIPSYCL_DEBUG_INFO << "### does not dominate:\n";
        HIPSYCL_DEBUG_EXECUTE_INFO(Ins.print(llvm::outs()); llvm::outs() << "\n";)
#endif
        llvm::StringRef BaseName;
        std::pair<llvm::StringRef, llvm::StringRef> Pieces = Operand->getName().rsplit('.');
        if (Pieces.second.startswith("pocl_"))
          BaseName = Pieces.first;
        else
          BaseName = Operand->getName();

        llvm::Value *Alternative = NULL;

        unsigned int CopyI = 0;
        do {
          std::ostringstream AlternativeName;
          AlternativeName << BaseName.str();
          if (CopyI > 0)
            AlternativeName << ".pocl_" << CopyI;

          Alternative = F.getValueSymbolTable()->lookup(AlternativeName.str());

          if (Alternative != NULL) {
            Ins.setOperand(Opr, Alternative);
            if (dominatesUse(DT, Ins, Opr))
              break;
          }

          if (CopyI > 10000 && Alternative == NULL)
            break; /* ran out of possibilities */
          ++CopyI;
        } while (true);

        if (Alternative != NULL) {
#ifdef DEBUG_REFERENCE_FIXING
          HIPSYCL_DEBUG_INFO << "### found the alternative:\n";
          HIPSYCL_DEBUG_EXECUTE_INFO(Alternative->print(llvm::outs()); llvm::outs() << "\n";)
#endif
          Changed |= true;
        } else {
#ifdef DEBUG_REFERENCE_FIXING
          HIPSYCL_DEBUG_INFO << "### didn't find an alternative for\n";
          HIPSYCL_DEBUG_EXECUTE_INFO(Operand->print(llvm::outs()); llvm::outs() << "\n";)
          HIPSYCL_DEBUG_INFO << "### BB:\n";
          HIPSYCL_DEBUG_EXECUTE_INFO(Operand->getParent()->print(llvm::outs()); llvm::outs() << "\n";)
          HIPSYCL_DEBUG_INFO << "### the user BB:\n";
          HIPSYCL_DEBUG_EXECUTE_INFO(Ins.getParent()->print(llvm::outs()); llvm::outs() << "\n";)
#endif
          llvm::outs().flush();
          std::cerr << "Could not find a dominating alternative variable." << std::endl;
          //          dumpCFG(F, "broken.dot");
          abort();
        }
      }
    }
  }
  return Changed;
}

} // namespace

namespace hipsycl::compiler {
char RealWorkItemLoopCreationPassLegacy::ID = 0;

void RealWorkItemLoopCreationPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::PostDominatorTreeWrapperPass>();

  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addRequired<llvm::DominatorTreeWrapperPass>();

  AU.addRequired<VariableUniformityAnalysisLegacy>();
  AU.addPreserved<VariableUniformityAnalysisLegacy>();

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool RealWorkItemLoopCreationPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  if (!SAA.isKernelFunc(&F) || utils::getSingleWorkItemLoop(LI))
    return false;

  auto &PDT = getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();
  auto &VUA = getAnalysis<VariableUniformityAnalysisLegacy>().getResult();

  WorkitemLoops WILC{F, DT, PDT, LI, VUA, SAA};
  bool Changed = WILC.processFunction(F);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  Changed |= WILC.fixUndominatedVariableUses(F);

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)

  return Changed;
}

llvm::PreservedAnalyses RealWorkItemLoopCreationPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
  if (!SAA || !SAA->isKernelFunc(&F) || utils::getSingleWorkItemLoop(LI)) {
    return llvm::PreservedAnalyses::all();
  }

  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);

  WorkitemLoops WILC{F, DT, PDT, LI, VUA, *SAA};

  bool Changed = WILC.processFunction(F);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  Changed |= WILC.fixUndominatedVariableUses(F);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)

  if (!Changed)
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<VariableUniformityAnalysis>();
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}

} // namespace hipsycl::compiler
