/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/compiler/cbs/SubCfgFormation.hpp"

#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/cbs/UniformityAnalysis.hpp"

#include "hipSYCL/common/debug.hpp"
#include "llvm/IR/GlobalVariable.h"

#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Regex.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/CodeExtractor.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

namespace {
using namespace hipsycl::compiler;

// Reference type only!
struct HierarchicalSplitInfo {
  bool IsSub;
  bool HasSub;
  llvm::ArrayRef<llvm::Value *> OuterLocalSize;
  llvm::ArrayRef<llvm::Value *> OuterIndices;
  llvm::Value *ContiguousIdx;
  llvm::Value *SGIdArg;
};

void formSubCfgsGeneric(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
                        const SplitterAnnotationInfo &SAA, llvm::Loop *WILoop, std::size_t Dim,
                        const std::size_t ReqdArrayElements, llvm::ArrayRef<llvm::Value *> LocalSize,
                        HierarchicalSplitInfo Hierarchy);

std::unique_ptr<hipsycl::compiler::RegionImpl> getRegion(llvm::Function &F, const llvm::LoopInfo &LI,
                                                         llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  if (auto *WILoop = utils::getSingleWorkItemLoop(LI))
    return std::unique_ptr<hipsycl::compiler::RegionImpl>{new hipsycl::compiler::LoopRegion(*WILoop)};
  else
    return std::unique_ptr<hipsycl::compiler::RegionImpl>{new hipsycl::compiler::FunctionRegion(F, Blocks)};
}
hipsycl::compiler::VectorizationInfo getVectorizationInfo(llvm::Function &F, hipsycl::compiler::Region &R,
                                                          llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                                                          llvm::PostDominatorTree &PDT, size_t Dim) {
  hipsycl::compiler::VectorizationInfo VecInfo{F, R};
  // seed varyingness
  if (auto *WILoop = utils::getSingleWorkItemLoop(LI)) {
    VecInfo.setPinnedShape(*WILoop->getCanonicalInductionVariable(), hipsycl::compiler::VectorShape::cont());
  } else {
    // todo: the work-group stuff is strided..?
    for (size_t D = 0; D < Dim - 1; ++D) {
      VecInfo.setPinnedShape(*utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[D]),
                             hipsycl::compiler::VectorShape::cont());
    }
    VecInfo.setPinnedShape(*utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1]),
                           hipsycl::compiler::VectorShape::cont());
    VecInfo.setPinnedShape(*utils::getLoadForGlobalVariable(F, SgIdGlobalName), hipsycl::compiler::VectorShape::cont());
  }

  hipsycl::compiler::VectorizationAnalysis VecAna{VecInfo, LI, DT, PDT};
  VecAna.analyze();
  return VecInfo;
}

void createLoopsAround(llvm::Function &F, llvm::BasicBlock *AfterBB, const llvm::ArrayRef<llvm::Value *> &LocalSize,
                       int EntryId, HierarchicalSplitInfo HI, llvm::ValueToValueMapTy &VMap,
                       llvm::SmallVector<llvm::BasicBlock *, 3> &Latches, llvm::BasicBlock *&LastHeader,
                       llvm::Value *&ContiguousIdx) {
  const auto &DL = F.getParent()->getDataLayout();
  auto *LoadBB = LastHeader;
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};

  const size_t Dim = LocalSize.size();
  llvm::SmallVector<llvm::PHINode *, 3> IndVars;
  for (int D = Dim - 1; D >= 0; --D) {
    const std::string Suffix = (llvm::Twine{HI.IsSub ? 's' : DimName[D]} + ".subcfg." + llvm::Twine{EntryId}).str();

    auto *Header = llvm::BasicBlock::Create(LastHeader->getContext(), "header." + Suffix + "b", LastHeader->getParent(),
                                            LastHeader);

    Builder.SetInsertPoint(Header, Header->getFirstInsertionPt());

    auto *WIIndVar = Builder.CreatePHI(DL.getLargestLegalIntType(F.getContext()), 2, "indvar." + Suffix);
    WIIndVar->addIncoming(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), 0), &F.getEntryBlock());
    IndVars.push_back(WIIndVar);
    Builder.CreateBr(LastHeader);

    auto *Latch = llvm::BasicBlock::Create(F.getContext(), "latch." + Suffix + "b", &F);
    Builder.SetInsertPoint(Latch, Latch->getFirstInsertionPt());

    llvm::Value *IncIndVar = Builder.CreateAdd(
        WIIndVar, Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), HI.HasSub && D == Dim - 1 ? SGSize : 1),
        "addInd." + Suffix, true, false);
    WIIndVar->addIncoming(IncIndVar, Latch);

    auto *LoopCond = Builder.CreateICmpULT(IncIndVar, LocalSize[D], "exit.cond." + Suffix);
    if (HI.IsSub && D == Dim - 1) {
      auto *ContCond = Builder.CreateICmpULT(ContiguousIdx, HI.OuterLocalSize.back(), "exit.cont_cond." + Suffix);
      LoopCond = Builder.CreateSelect(ContCond, LoopCond, llvm::ConstantInt::getNullValue(LoopCond->getType()));
    }
    Builder.CreateCondBr(LoopCond, Header, AfterBB);

    Latches.push_back(Latch);
    LastHeader = Header;
  }

  std::reverse(Latches.begin(), Latches.end());
  std::reverse(IndVars.begin(), IndVars.end());

  for (size_t D = 1; D < Dim; ++D) {
    Latches[D]->getTerminator()->replaceSuccessorWith(AfterBB, Latches[D - 1]);
    IndVars[D]->replaceIncomingBlockWith(&F.getEntryBlock(), IndVars[D - 1]->getParent());
  }

  if (!HI.HasSub || HI.IsSub) {
    auto *MDWorkItemLoop =
        llvm::MDNode::get(F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::WorkItemLoop)});
    auto *LoopID = llvm::makePostTransformationMetadata(F.getContext(), nullptr, {}, {MDWorkItemLoop});
    Latches[Dim - 1]->getTerminator()->setMetadata("llvm.loop", LoopID);
  }
  VMap[AfterBB] = Latches[Dim - 1];

  Builder.SetInsertPoint(IndVars[Dim - 1]->getParent(), ++IndVars[Dim - 1]->getIterator());
  llvm::Value *Idx = IndVars[0];
  if (!HI.IsSub) {
    for (size_t D = 1; D < Dim; ++D) {
      const std::string Suffix = (llvm::Twine{DimName[D]} + ".subcfg." + llvm::Twine{EntryId}).str();

      Idx = Builder.CreateMul(Idx, LocalSize[D], "idx.mul." + Suffix, true);
      Idx = Builder.CreateAdd(IndVars[D], Idx, "idx.add." + Suffix, true);

      if (!HI.HasSub || D != Dim - 1)
        VMap[utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[D])] = IndVars[D];
    }
    if (!HI.HasSub) {
      Builder.SetInsertPoint(LoadBB, LoadBB->getFirstInsertionPt());
      VMap[utils::getLoadForGlobalVariable(F, SgIdGlobalName)] =
          Builder.CreateURem(IndVars.back(), llvm::ConstantInt::get(IndVars.back()->getType(), SGSize));
    }
  } else {
    VMap[utils::getLoadForGlobalVariable(F, SgIdGlobalName)] = Idx;
    VMap[HI.SGIdArg] = Idx;
    Builder.SetInsertPoint(LoadBB, LoadBB->getFirstInsertionPt());
    auto StridedInner = llvm::cast<llvm::Instruction>(HI.OuterIndices.back())->getOperand(0);
    Idx = Builder.CreateAdd(Idx, StridedInner);
    // fixme: this is not actually the contiguous index for multi-dim..
  }

  // todo: replace `ret` with branch to innermost latch

  if (!HI.HasSub || Dim != 1)
    VMap[utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[0])] = IndVars[0];
  else if (HI.IsSub)
    VMap[utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])] = Idx;
  else {
    Builder.SetInsertPoint(LoadBB, LoadBB->getFirstInsertionPt());
    VMap[utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])] =
        Builder.CreateAdd(IndVars.back(), utils::getLoadForGlobalVariable(F, SgIdGlobalName));
  }
  if (!HI.HasSub) {
    VMap[ContiguousIdx] = Idx;
    ContiguousIdx = Idx;
  } else
    VMap[ContiguousIdx] = ContiguousIdx;

  if (HI.IsSub) {
    llvm::ValueToValueMapTy VMap;
    VMap[HI.ContiguousIdx] = Idx;
    VMap[utils::getLoadForGlobalVariable(F, SgIdGlobalName)] = IndVars[0];
    VMap[HI.SGIdArg] = IndVars[0];
    llvm::SmallVector<llvm::BasicBlock *> Blocks{Latches.begin(), Latches.end()};
    Blocks.push_back(LoadBB);
    llvm::remapInstructionsInBlocks(Blocks, VMap);
  }
}

class SubCFG {

  using BlockVector = llvm::SmallVector<llvm::BasicBlock *, 8>;
  BlockVector Blocks_;
  BlockVector NewBlocks_;
  size_t EntryId_;
  llvm::BasicBlock *EntryBarrier_;
  llvm::SmallDenseMap<llvm::BasicBlock *, size_t> ExitIds_;
  llvm::AllocaInst *LastBarrierIdStorage_;
  llvm::BasicBlock *EntryBB_;
  llvm::BasicBlock *ExitBB_;
  llvm::BasicBlock *LoadBB_;
  llvm::BasicBlock *PreHeader_;
  llvm::SmallVector<llvm::Value *, 3> WIIndVars_;
  size_t Dim;
  HierarchicalSplitInfo HI;

  //  void addBlock(llvm::BasicBlock *BB) { Blocks_.push_back(BB); }
  llvm::BasicBlock *createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, unsigned long> BarrierPair,
                                     llvm::BasicBlock *After, llvm::BasicBlock *WILatch);

  void loadMultiSubCfgValues(
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap);
  llvm::BasicBlock *createLoadBB(llvm::ValueToValueMapTy &VMap);
  llvm::BasicBlock *createUniformLoadBB(llvm::BasicBlock *OuterMostHeader);

  llvm::SmallVector<llvm::Instruction *, 16>
  topoSortInstructions(const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const;

public:
  SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
         const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds, const llvm::Loop *WILoop,
         const SplitterAnnotationInfo &SAA, size_t Dim, HierarchicalSplitInfo HI);

  SubCFG(const SubCFG &) = delete;
  SubCFG &operator=(const SubCFG &) = delete;

  SubCFG(SubCFG &&) = default;
  SubCFG &operator=(SubCFG &&) = default;

  BlockVector &getBlocks() noexcept { return Blocks_; }
  const BlockVector &getBlocks() const noexcept { return Blocks_; }

  BlockVector &getNewBlocks() noexcept { return NewBlocks_; }
  const BlockVector &getNewBlocks() const noexcept { return NewBlocks_; }

  size_t getEntryId() const noexcept { return EntryId_; }

  llvm::BasicBlock *getEntry() noexcept { return EntryBB_; }
  llvm::BasicBlock *getExit() noexcept { return ExitBB_; }
  llvm::BasicBlock *getLoadBB() noexcept { return LoadBB_; }
  llvm::Value *getWIIndVar() noexcept { return HI.ContiguousIdx; }
  const llvm::SmallVector<llvm::Value *, 3> &getWIIndVars() const noexcept { return WIIndVars_; }
  const HierarchicalSplitInfo &getHI() const noexcept { return HI; }

  void replicate(llvm::Loop *WILoop, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap);
  void replicate(llvm::Function &WILoop, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
                 llvm::BasicBlock *AfterBB, llvm::ArrayRef<llvm::Value *> LocalSize, const SplitterAnnotationInfo &SAA);

  void arrayifyMultiSubCfgValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
      llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, size_t ReqdArrayElements,
      hipsycl::compiler::VectorizationInfo &VecInfo);
  void fixSingleSubCfgValues(llvm::DominatorTree &DT,
                             const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
                             std::size_t ReqdArrayElements, hipsycl::compiler::VectorizationInfo &VecInfo);

  void print() const;
  void removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const;
};

llvm::BasicBlock *SubCFG::createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
                                           llvm::BasicBlock *After, llvm::BasicBlock *WILatch) {
  HIPSYCL_DEBUG_INFO << "Create new exit with ID: " << BarrierPair.second << " at " << After->getName() << "\n";

  auto *Exit = llvm::BasicBlock::Create(After->getContext(),
                                        After->getName() + ".subcfg.exit" + llvm::Twine{BarrierPair.second} + "b",
                                        After->getParent(), WILatch);

  auto &DL = Exit->getParent()->getParent()->getDataLayout();
  llvm::IRBuilder Builder{Exit, Exit->getFirstInsertionPt()};
  Builder.CreateStore(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), BarrierPair.second),
                      LastBarrierIdStorage_);
  Builder.CreateBr(WILatch);

  After->getTerminator()->replaceSuccessorWith(BarrierPair.first, Exit);
  return Exit;
}

SubCFG::SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
               const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds, const llvm::Loop *WILoop,
               const SplitterAnnotationInfo &SAA, size_t Dim, HierarchicalSplitInfo HI)
    : LastBarrierIdStorage_(LastBarrierIdStorage), EntryId_(BarrierIds.lookup(EntryBarrier)),
      EntryBarrier_(EntryBarrier), EntryBB_(EntryBarrier->getSingleSuccessor()), LoadBB_(nullptr), PreHeader_(nullptr),
      Dim(Dim), HI(HI) {
  const auto *WILatch = WILoop ? WILoop->getLoopLatch() : nullptr;

  assert(HI.ContiguousIdx && "Must have found either IndVar or __hipsycl_local_id_{x,y,z}");

  llvm::SmallVector<llvm::BasicBlock *, 4> WL{EntryBarrier};
  while (!WL.empty()) {
    auto *BB = WL.pop_back_val();

    llvm::SmallVector<llvm::BasicBlock *, 2> Succs{llvm::succ_begin(BB), llvm::succ_end(BB)};
    for (auto *Succ : Succs) {
      if (WILatch == Succ || std::find(Blocks_.begin(), Blocks_.end(), Succ) != Blocks_.end())
        continue;

      if (!(HI.IsSub ? utils::hasOnlySubBarrier(Succ, SAA) : utils::hasOnlyBarrier(Succ, SAA))) {
        WL.push_back(Succ);
        Blocks_.push_back(Succ);
      } else {
        size_t BId = BarrierIds.lookup(Succ);
        assert(BId != 0 && "Exit barrier block not found in map");
        ExitIds_.insert({Succ, BId});
      }
    }
  }
}

void SubCFG::print() const {
  HIPSYCL_DEBUG_INFO << "SubCFG entry barrier: " << EntryId_ << "\n";
  HIPSYCL_DEBUG_INFO << "SubCFG block names: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *BB : Blocks_) { llvm::outs() << BB->getName() << ", "; } llvm::outs() << "\n";)
  HIPSYCL_DEBUG_INFO << "SubCFG exits: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto ExitIt
                                  : ExitIds_) {
    llvm::outs() << ExitIt.first->getName() << " (" << ExitIt.second << "), ";
  } llvm::outs() << "\n";)
  HIPSYCL_DEBUG_INFO << "SubCFG new block names: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *BB
                                  : NewBlocks_) {
    llvm::outs() << BB->getName() << ", ";
  } llvm::outs() << "\n";)
}

void addRemappedDenseMapKeys(const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &OrgInstAllocaMap,
                             const llvm::ValueToValueMapTy &VMap,
                             llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &NewInstAllocaMap) {
  for (auto &InstAllocaPair : OrgInstAllocaMap) {
    if (auto *NewInst = llvm::dyn_cast_or_null<llvm::Instruction>(VMap.lookup(InstAllocaPair.first)))
      NewInstAllocaMap.insert({NewInst, InstAllocaPair.second});
  }
}

void SubCFG::replicate(
    llvm::Loop *WILoop, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap) {
  llvm::ValueToValueMapTy VMap;
  auto *OrgWIPreHeader = WILoop->getLoopPreheader();

  auto *WIHeader = llvm::CloneBasicBlock(WILoop->getHeader(), VMap, ".subcfg." + llvm::Twine{EntryId_} + "b",
                                         WILoop->getHeader()->getParent());
  auto *WILatch = llvm::CloneBasicBlock(WILoop->getLoopLatch(), VMap, ".subcfg." + llvm::Twine{EntryId_} + "b",
                                        WIHeader->getParent());

  VMap[WILoop->getHeader()] = WIHeader;
  VMap[WILoop->getLoopLatch()] = WILatch;

  for (auto *BB : Blocks_) {
    auto *NewBB = llvm::CloneBasicBlock(BB, VMap, ".subcfg." + llvm::Twine{EntryId_} + "b", WIHeader->getParent());
    VMap[BB] = NewBB;
    NewBlocks_.push_back(NewBB);
    for (auto *Succ : llvm::successors(BB)) {
      if (auto ExitIt = ExitIds_.find(Succ); ExitIt != ExitIds_.end()) {
        NewBlocks_.push_back(createExitWithID(*ExitIt, NewBB, WILatch));
      }
    }
  }
  print();

  addRemappedDenseMapKeys(InstAllocaMap, VMap, RemappedInstAllocaMap);
  LoadBB_ = createLoadBB(VMap);
  VMap[EntryBarrier_] = LoadBB_;
  PreHeader_ = createUniformLoadBB(LoadBB_);
  WIHeader->replacePhiUsesWith(OrgWIPreHeader, PreHeader_);

  loadMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap);
  VMap[utils::getWorkItemLoopBodyEntry(WILoop)] = LoadBB_;

  llvm::SmallVector<llvm::BasicBlock *, 8> BlocksToRemap{NewBlocks_.begin(), NewBlocks_.end()};
  BlocksToRemap.push_back(WIHeader);
  BlocksToRemap.push_back(WILatch);
  llvm::remapInstructionsInBlocks(BlocksToRemap, VMap);

  if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(VMap[HI.ContiguousIdx])) {
    auto *LatchV = Phi->getIncomingValueForBlock(WILatch);

    for (auto *U : Phi->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (UI->getParent() == WIHeader)
          UI->replaceUsesOfWith(Phi, LatchV);
      }
    }

    // Move PHI from Header to for body
    Phi->moveBefore(&*LoadBB_->begin());
    Phi->replaceIncomingBlockWith(WILatch, WIHeader);
    VMap[WILoop->getHeader()] = LoadBB_;
  }

  // Header is now latch, so copy loop md over
  WIHeader->getTerminator()->setMetadata("llvm.loop", WILatch->getTerminator()->getMetadata("llvm.loop"));
  WILatch->getTerminator()->setMetadata("llvm.loop", nullptr);

  removeDeadPhiBlocks(BlocksToRemap);

  EntryBB_ = PreHeader_;
  ExitBB_ = WIHeader;
  HI.ContiguousIdx = VMap[HI.ContiguousIdx];
}

void SubCFG::replicate(
    llvm::Function &F, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap, llvm::BasicBlock *AfterBB,
    llvm::ArrayRef<llvm::Value *> LocalSize, const SplitterAnnotationInfo &SAA) {
  auto &DL = F.getParent()->getDataLayout();
  llvm::ValueToValueMapTy VMap;

  for (auto *BB : Blocks_) {
    auto *NewBB = llvm::CloneBasicBlock(BB, VMap, ".subcfg." + llvm::Twine{EntryId_} + "b", &F);
    VMap[BB] = NewBB;
    NewBlocks_.push_back(NewBB);
    for (auto *Succ : llvm::successors(BB)) {
      if (auto ExitIt = ExitIds_.find(Succ); ExitIt != ExitIds_.end()) {
        NewBlocks_.push_back(createExitWithID(*ExitIt, NewBB, AfterBB));
      }
    }
  }

  LoadBB_ = createLoadBB(VMap);

  VMap[EntryBarrier_] = LoadBB_;

  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  llvm::BasicBlock *LastHeader = LoadBB_;
  llvm::Value *Idx = HI.ContiguousIdx;

  createLoopsAround(F, AfterBB, LocalSize, EntryId_, HI, VMap, Latches, LastHeader, Idx);
  for (size_t D = 0; D < LocalSize.size(); ++D) {
    WIIndVars_.push_back(VMap[HI.IsSub ? HI.SGIdArg : utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[D])]);
  }

  PreHeader_ = createUniformLoadBB(LastHeader);
  LastHeader->replacePhiUsesWith(&F.getEntryBlock(), PreHeader_);

  print();

  addRemappedDenseMapKeys(InstAllocaMap, VMap, RemappedInstAllocaMap);
  loadMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap);

  llvm::SmallVector<llvm::BasicBlock *, 8> BlocksToRemap{NewBlocks_.begin(), NewBlocks_.end()};
  llvm::remapInstructionsInBlocks(BlocksToRemap, VMap);

  removeDeadPhiBlocks(BlocksToRemap);

  EntryBB_ = PreHeader_;
  ExitBB_ = Latches[0];
  HI.ContiguousIdx = Idx;
  HI.SGIdArg = WIIndVars_.back();
}

void SubCFG::removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const {
  for (auto *BB : BlocksToRemap) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 4> Predecessors{llvm::pred_begin(BB), llvm::pred_end(BB)};
    for (auto &I : *BB) {
      if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&I)) {
        llvm::SmallVector<llvm::BasicBlock *, 4> IncomingBlocksToRemove;
        for (int IncomingIdx = 0; IncomingIdx < Phi->getNumIncomingValues(); ++IncomingIdx) {
          auto *IncomingBB = Phi->getIncomingBlock(IncomingIdx);
          if (!Predecessors.contains(IncomingBB))
            IncomingBlocksToRemove.push_back(IncomingBB);
        }
        for (auto *IncomingBB : IncomingBlocksToRemove) {
          HIPSYCL_DEBUG_INFO << "[SubCFG] Remove incoming block " << IncomingBB->getName() << " from PHI " << *Phi
                             << "\n";
          Phi->removeIncomingValue(IncomingBB);
          HIPSYCL_DEBUG_INFO << "[SubCFG] Removed incoming block " << IncomingBB->getName() << " from PHI " << *Phi
                             << "\n";
        }
      }
    }
  }
}

bool dontArrayifyContiguousValues(
    llvm::Instruction &I, llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::Instruction *AllocaIP, size_t ReqdArrayElements, llvm::Value *IndVar,
    hipsycl::compiler::VectorizationInfo &VecInfo) {
  if (VecInfo.isPinned(I))
    return true;

  llvm::SmallVector<llvm::Instruction *, 4> WL;
  llvm::SmallPtrSet<llvm::Instruction *, 8> UniformValues;
  llvm::SmallVector<llvm::Instruction *, 8> ContiguousInsts;
  llvm::SmallPtrSet<llvm::Value *, 8> LookedAt;
  HIPSYCL_DEBUG_INFO << "[SubCFG] IndVar: " << *IndVar << "\n";
  WL.push_back(&I);
  while (!WL.empty()) {
    auto *WLValue = WL.pop_back_val();
    if (auto *WLI = llvm::dyn_cast<llvm::Instruction>(WLValue))
      for (auto *V : WLI->operand_values()) {
        HIPSYCL_DEBUG_INFO << "[SubCFG] Considering: " << *V << "\n";

        if (V == IndVar || VecInfo.isPinned(*V))
          continue;
        // todo: fix PHIs
        if (!LookedAt.insert(V).second)
          return false;
        if (auto *OpI = llvm::dyn_cast<llvm::Instruction>(V)) {
          if (VecInfo.getVectorShape(*OpI).isContiguous()) {
            WL.push_back(OpI);
            ContiguousInsts.push_back(OpI);
          } else if (!UniformValues.contains(OpI))
            UniformValues.insert(OpI);
        }
      }
  }
  for (auto *UI : UniformValues) {
    HIPSYCL_DEBUG_INFO << "[SubCFG] UniValue to store: " << *UI << "\n";
    if (BaseInstAllocaMap.lookup(UI))
      continue;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Store required uniform value to single element alloca " << I << "\n";
    auto *Alloca = utils::arrayifyInstruction(AllocaIP, UI, IndVar, 1);
    BaseInstAllocaMap.insert({UI, Alloca});
    VecInfo.setVectorShape(*Alloca, hipsycl::compiler::VectorShape::uni());
  }
  ContInstReplicaMap.insert({&I, ContiguousInsts});
  return true;
}

void SubCFG::arrayifyMultiSubCfgValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, size_t ReqdArrayElements,
    hipsycl::compiler::VectorizationInfo &VecInfo) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> OtherCFGBlocks;
  for (auto &Cfg : SubCFGs) {
    if (&Cfg != this)
      OtherCFGBlocks.insert(Cfg.Blocks_.begin(), Cfg.Blocks_.end());
  }

  auto ContiguousIdx = HI.IsSub ? HI.SGIdArg : HI.ContiguousIdx;

  for (auto *BB : Blocks_) {
    for (auto &I : *BB) {
      if (&I == ContiguousIdx)
        continue;
      if (InstAllocaMap.lookup(&I))
        continue;
      if (utils::anyOfUsers<llvm::Instruction>(&I, [&OtherCFGBlocks, this, &I](auto *UI) {
            return UI->getParent() != I.getParent() && OtherCFGBlocks.contains(UI->getParent());
          })) {
        if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(&I))
          if (auto *Alloca = utils::getLoopStateAllocaForLoad(*LInst)) {
            InstAllocaMap.insert({&I, Alloca});
            continue;
          }
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
          if (GEP->hasMetadata(hipsycl::compiler::MDKind::Arrayified)) {
            // fixme: in the sub path, this might be a function argument.. currently only triggers on debug though.
            InstAllocaMap.insert({&I, llvm::cast<llvm::AllocaInst>(GEP->getPointerOperand())});
            continue;
          }

        auto Shape = VecInfo.getVectorShape(I);
#ifndef HIPSYCL_NO_PHIS_IN_SPLIT
        if (Shape.isUniform()) {
          HIPSYCL_DEBUG_INFO << "[SubCFG] Value uniform, store to single element alloca " << I << "\n";
          auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, ContiguousIdx, 1);
          InstAllocaMap.insert({&I, Alloca});
          VecInfo.setVectorShape(*Alloca, hipsycl::compiler::VectorShape::uni());
          continue;
        }
#endif
#ifndef HIPSYCL_NO_CONTIGUOUS_VALUES
        if (Shape.isContiguous()) {
          if (dontArrayifyContiguousValues(I, BaseInstAllocaMap, ContInstReplicaMap, AllocaIP, ReqdArrayElements,
                                           ContiguousIdx, VecInfo)) {
            HIPSYCL_DEBUG_INFO << "[SubCFG] Not arrayifying " << I << "\n";
            continue;
          }
        }
#endif
        auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, ContiguousIdx, ReqdArrayElements);
        InstAllocaMap.insert({&I, Alloca});
        VecInfo.setVectorShape(*Alloca, Shape);
      }
    }
  }
}

void remapInstruction(llvm::Instruction *I, llvm::ValueToValueMapTy &VMap) {
  llvm::SmallVector<llvm::Value *, 8> WL{I->value_op_begin(), I->value_op_end()};
  for (auto *V : WL) {
    if (VMap.count(V))
      I->replaceUsesOfWith(V, VMap[V]);
  }
  HIPSYCL_DEBUG_INFO << "[SubCFG] remapped Inst " << *I << "\n";
}

void SubCFG::loadMultiSubCfgValues(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap) {
  llvm::Value *NewWIIndVar = HI.IsSub ? VMap[HI.SGIdArg] : VMap[HI.ContiguousIdx];
  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  llvm::IRBuilder Builder{LoadTerm};

  for (auto &InstAllocaPair : InstAllocaMap) {
    if (std::find(Blocks_.begin(), Blocks_.end(), InstAllocaPair.first->getParent()) == Blocks_.end()) {
      if (utils::anyOfUsers<llvm::Instruction>(InstAllocaPair.first, [this](llvm::Instruction *UI) {
            return std::find(NewBlocks_.begin(), NewBlocks_.end(), UI->getParent()) != NewBlocks_.end();
          })) {
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(InstAllocaPair.first))
          if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
            auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(
                Builder.CreateInBoundsGEP(GEP->getType(), GEP->getPointerOperand(), NewWIIndVar, GEP->getName() + "c"));
            NewGEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDArrayified);
            VMap[InstAllocaPair.first] = NewGEP;
            continue;
          }
        auto *IP = LoadTerm;
        if (!InstAllocaPair.second->isArrayAllocation())
          IP = UniformLoadTerm;
        HIPSYCL_DEBUG_INFO << "[SubCFG] Load from Alloca " << *InstAllocaPair.second << " in "
                           << IP->getParent()->getName() << "\n";
        auto *Load = utils::loadFromAlloca(InstAllocaPair.second, NewWIIndVar, IP, InstAllocaPair.first->getName());
        utils::copyDgbValues(InstAllocaPair.first, Load, IP);
        VMap[InstAllocaPair.first] = Load;
      }
    }
  }

  llvm::ValueToValueMapTy UniVMap;
  UniVMap[this->HI.ContiguousIdx] = VMap[HI.ContiguousIdx];

  // copy local id load value to univmap
  for (size_t D = 0; D < this->Dim; ++D) {
    auto *Load =
        HI.IsSub ? HI.SGIdArg : utils::getLoadForGlobalVariable(*this->LoadBB_->getParent(), LocalIdGlobalNames[D]);
    UniVMap[Load] = VMap[Load];
  }

  // load uniform values from allocas
  for (auto &InstAllocaPair : BaseInstAllocaMap) {
    auto *IP = UniformLoadTerm;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Load base value from Alloca " << *InstAllocaPair.second << " in "
                       << IP->getParent()->getName() << "\n";
    auto *Load = utils::loadFromAlloca(InstAllocaPair.second, NewWIIndVar, IP, InstAllocaPair.first->getName());
    utils::copyDgbValues(InstAllocaPair.first, Load, IP);
    UniVMap[InstAllocaPair.first] = Load;
  }

  // get a set of unique contiguous instructions
  llvm::SmallPtrSet<llvm::Instruction *, 16> UniquifyInsts;
  for (auto &Pair : ContInstReplicaMap) {
    UniquifyInsts.insert(Pair.first);
    for (auto &Target : Pair.second)
      UniquifyInsts.insert(Target);
  }

  auto OrderedInsts = topoSortInstructions(UniquifyInsts);

  llvm::SmallPtrSet<llvm::Instruction *, 16> InstsToRemap;
  // clone the contiguous instructions to restore the used values
  for (auto *I : OrderedInsts) {
    if (UniVMap.count(I))
      continue;

    HIPSYCL_DEBUG_INFO << "[SubCFG] Clone cont instruction and operands of: " << *I << " to "
                       << LoadTerm->getParent()->getName() << "\n";
    auto *IClone = I->clone();
    IClone->insertBefore(LoadTerm);
    InstsToRemap.insert(IClone);
    UniVMap[I] = IClone;
    if (VMap.count(I) == 0)
      VMap[I] = IClone;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Clone cont instruction: " << *IClone << "\n";
  }

  // finally remap the singular instructions to use the other cloned contiguous instructions /
  // uniform values
  for (auto *IToRemap : InstsToRemap)
    remapInstruction(IToRemap, UniVMap);
}

llvm::SmallVector<llvm::Instruction *, 16>
SubCFG::topoSortInstructions(const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const {
  llvm::SmallVector<llvm::Instruction *, 16> OrderedInsts(UniquifyInsts.size());
  std::copy(UniquifyInsts.begin(), UniquifyInsts.end(), OrderedInsts.begin());

  auto IsUsedBy = [](llvm::Instruction *LHS, llvm::Instruction *RHS) {
    for (auto *U : LHS->users()) {
      if (U == RHS)
        return true;
    }
    return false;
  };
  for (int I = 0; I < OrderedInsts.size(); ++I) {
    int InsertAt = I;
    for (int J = OrderedInsts.size() - 1; J > I; --J) {
      if (IsUsedBy(OrderedInsts[J], OrderedInsts[I])) {
        InsertAt = J;
        break;
      }
    }
    if (InsertAt != I) {
      auto *Tmp = OrderedInsts[I];
      for (int J = I + 1; J <= InsertAt; ++J) {
        OrderedInsts[J - 1] = OrderedInsts[J];
      }
      OrderedInsts[InsertAt] = Tmp;
      --I;
    }
  }
  return OrderedInsts;
}

llvm::BasicBlock *SubCFG::createUniformLoadBB(llvm::BasicBlock *OuterMostHeader) {
  auto *LoadBB =
      llvm::BasicBlock::Create(OuterMostHeader->getContext(), "uniloadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                               OuterMostHeader->getParent(), OuterMostHeader);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(OuterMostHeader);
  return LoadBB;
}

llvm::BasicBlock *SubCFG::createLoadBB(llvm::ValueToValueMapTy &VMap) {
  auto *NewEntry = llvm::cast<llvm::BasicBlock>(static_cast<llvm::Value *>(VMap[EntryBB_]));
  auto *LoadBB = llvm::BasicBlock::Create(NewEntry->getContext(), "loadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                                          NewEntry->getParent(), NewEntry);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(NewEntry);
  return LoadBB;
}

void SubCFG::fixSingleSubCfgValues(llvm::DominatorTree &DT,
                                   const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
                                   std::size_t ReqdArrayElements, hipsycl::compiler::VectorizationInfo &VecInfo) {

  auto *AllocaIP = LoadBB_->getParent()->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
  auto *LoadIP = LoadBB_->getTerminator();
  auto *UniLoadIP = PreHeader_->getTerminator();
  llvm::IRBuilder Builder{LoadIP};

  llvm::DenseMap<llvm::Instruction *, llvm::Instruction *> InstLoadMap;
  llvm::Value *ContiguousIdx = HI.IsSub ? HI.SGIdArg : HI.ContiguousIdx;

  for (auto *BB : NewBlocks_) {
    llvm::SmallVector<llvm::Instruction *, 16> Insts{};
    std::transform(BB->begin(), BB->end(), std::back_inserter(Insts), [](auto &I) { return &I; });
    for (auto *Inst : Insts) {
      auto &I = *Inst;
      for (auto *OPV : I.operand_values()) {
        if (auto *OPI = llvm::dyn_cast<llvm::Instruction>(OPV); OPI && !DT.dominates(OPI, &I)) {
          if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(Inst)) {
            bool FoundIncoming = false;
            for (auto &Incoming : Phi->incoming_values()) {
              if (OPV == Incoming.get()) {
                auto *IncomingBB = Phi->getIncomingBlock(Incoming);
                if (DT.dominates(OPI, IncomingBB->getTerminator())) {
                  FoundIncoming = true;
                  break;
                }
              }
            }
            if (FoundIncoming)
              continue;
          }
          HIPSYCL_DEBUG_WARNING << "Instruction not dominated " << I << " operand: " << *OPI << "\n";

          if (auto *Load = InstLoadMap.lookup(OPI))
            // if the already inserted Load does not dominate I, we must create another load.
            if (DT.dominates(Load, &I)) {
              I.replaceUsesOfWith(OPI, Load);
              continue;
            }

          if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(OPI))
            if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
              auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(Builder.CreateInBoundsGEP(
                  GEP->getType(), GEP->getPointerOperand(), ContiguousIdx, GEP->getName() + "c"));
              NewGEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDArrayified);
              I.replaceUsesOfWith(OPI, NewGEP);
              InstLoadMap.insert({OPI, NewGEP});
              continue;
            }

          llvm::AllocaInst *Alloca = nullptr;
          if (auto *RemAlloca = RemappedInstAllocaMap.lookup(OPI))
            Alloca = RemAlloca;
          if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(OPI))
            Alloca = utils::getLoopStateAllocaForLoad(*LInst);
          if (!Alloca) {
            HIPSYCL_DEBUG_INFO << "[SubCFG] No alloca, yet for " << *OPI << "\n";
            //            if (VecInfo.getVectorShape(I).isUniform())
            //              Alloca = utils::arrayifyInstruction(AllocaIP, OPI, ContiguousIdx, 1);
            //            else
            Alloca = utils::arrayifyInstruction(AllocaIP, OPI, ContiguousIdx, ReqdArrayElements);
            VecInfo.setVectorShape(*Alloca, VecInfo.getVectorShape(I));
          }

#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          // in split loop, OPI might be used multiple times, get the user, dominating this user and insert load there
          llvm::Instruction *NewIP = &I;
          for (auto *U : OPI->users()) {
            if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U); UI && DT.dominates(UI, NewIP)) {
              NewIP = UI;
            }
          }
#else
          auto *NewIP = LoadIP;
          if (!Alloca->isArrayAllocation())
            NewIP = UniLoadIP;
#endif

          auto *Load = utils::loadFromAlloca(Alloca, ContiguousIdx, NewIP, OPI->getName());
          utils::copyDgbValues(OPI, Load, NewIP);

#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          I.replaceUsesOfWith(OPI, Load);
          InstLoadMap.insert({OPI, Load});
#else
          const auto NumPreds = std::distance(llvm::pred_begin(BB), llvm::pred_end(BB));
          if (!llvm::isa<llvm::PHINode>(I) && NumPreds > 1 &&
              std::find(llvm::pred_begin(BB), llvm::pred_end(BB), LoadBB_) != llvm::pred_end(BB)) {
            Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
            auto *PHINode = Builder.CreatePHI(Load->getType(), NumPreds, I.getName());
            for (auto *PredBB : llvm::predecessors(BB))
              if (PredBB == LoadBB_)
                PHINode->addIncoming(Load, PredBB);
              else
                PHINode->addIncoming(OPV, PredBB);

            I.replaceUsesOfWith(OPI, PHINode);
            InstLoadMap.insert({OPI, PHINode});
          } else {
            I.replaceUsesOfWith(OPI, Load);
            InstLoadMap.insert({OPI, Load});
          }
#endif
        }
      }
    }
  }
}

llvm::BasicBlock *createUnreachableBlock(llvm::Function &F) {
  auto *Default = llvm::BasicBlock::Create(F.getContext(), "cbs.while.default", &F);
  llvm::IRBuilder Builder{Default, Default->getFirstInsertionPt()};
  Builder.CreateUnreachable();
  return Default;
}

llvm::BasicBlock *generateWhileSwitchAround(llvm::BasicBlock *PreHeader, llvm::BasicBlock *OldEntry,
                                            llvm::BasicBlock *Exit, llvm::AllocaInst *LastBarrierIdStorage,
                                            std::vector<SubCFG> &SubCFGs) {
  auto &F = *PreHeader->getParent();
  auto &M = *F.getParent();
  const auto &DL = M.getDataLayout();

  auto *WhileHeader =
      llvm::BasicBlock::Create(PreHeader->getContext(), "cbs.while.header", PreHeader->getParent(), OldEntry);
  llvm::IRBuilder Builder{WhileHeader, WhileHeader->getFirstInsertionPt()};
  auto *LastID =
      Builder.CreateLoad(LastBarrierIdStorage->getAllocatedType(), LastBarrierIdStorage, "cbs.while.last_barr.load");
  auto *Switch = Builder.CreateSwitch(LastID, createUnreachableBlock(F), SubCFGs.size());
  for (auto &Cfg : SubCFGs) {
    Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), Cfg.getEntryId()), Cfg.getEntry());
    Cfg.getEntry()->replacePhiUsesWith(PreHeader, WhileHeader);
    Cfg.getExit()->getTerminator()->replaceSuccessorWith(Exit, WhileHeader);
  }
  Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), ExitBarrierId), Exit);

  Builder.SetInsertPoint(PreHeader->getTerminator());
  Builder.CreateStore(llvm::ConstantInt::get(LastBarrierIdStorage->getAllocatedType(), EntryBarrierId),
                      LastBarrierIdStorage);
  PreHeader->getTerminator()->replaceSuccessorWith(OldEntry, WhileHeader);
  return WhileHeader;
}

void purgeLifetime(SubCFG &Cfg) {
  llvm::SmallVector<llvm::Instruction *, 8> ToDelete;
  for (auto *BB : Cfg.getNewBlocks())
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

void fillUserHull(llvm::AllocaInst *Alloca, llvm::SmallVectorImpl<llvm::Instruction *> &Hull) {
  llvm::SmallVector<llvm::Instruction *, 8> WL;
  std::transform(Alloca->user_begin(), Alloca->user_end(), std::back_inserter(WL),
                 [](auto *U) { return llvm::cast<llvm::Instruction>(U); });
  llvm::SmallPtrSet<llvm::Instruction *, 32> AlreadySeen;
  while (!WL.empty()) {
    auto *I = WL.pop_back_val();
    AlreadySeen.insert(I);
    Hull.push_back(I);
    for (auto *U : I->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (!AlreadySeen.contains(UI))
          if (UI->mayReadOrWriteMemory() || UI->getType()->isPointerTy())
            WL.push_back(UI);
      }
    }
  }
}

bool isAllocaSubCfgInternal(llvm::AllocaInst *Alloca, const std::vector<SubCFG> &SubCfgs,
                            const llvm::DominatorTree &DT) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> UserBlocks;
  {
    llvm::SmallVector<llvm::Instruction *, 32> Users;
    fillUserHull(Alloca, Users);
    utils::PtrSetWrapper<decltype(UserBlocks)> Wrapper{UserBlocks};
    std::transform(Users.begin(), Users.end(), std::inserter(Wrapper, UserBlocks.end()),
                   [](auto *I) { return I->getParent(); });
  }

  for (auto &SubCfg : SubCfgs) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 8> SubCfgSet{SubCfg.getNewBlocks().begin(), SubCfg.getNewBlocks().end()};
    if (std::any_of(UserBlocks.begin(), UserBlocks.end(), [&SubCfgSet](auto *BB) { return SubCfgSet.contains(BB); }) &&
        !std::all_of(UserBlocks.begin(), UserBlocks.end(), [&SubCfgSet, Alloca](auto *BB) {
          if (SubCfgSet.contains(BB)) {
            return true;
          }
          HIPSYCL_DEBUG_INFO << "[SubCFG] BB not in subcfgset: " << BB->getName() << " for alloca: ";
          HIPSYCL_DEBUG_EXECUTE_INFO(Alloca->print(llvm::outs()); llvm::outs() << "\n";)
          return false;
        }))
      return false;
  }

  return true;
}

void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT, std::vector<SubCFG> &SubCfgs,
                     std::size_t ReqdArrayElements, hipsycl::compiler::VectorizationInfo &VecInfo) {
  auto *MDAlloca =
      llvm::MDNode::get(EntryBlock->getContext(), {llvm::MDString::get(EntryBlock->getContext(), "hipSYCLLoopState")});

  llvm::SmallPtrSet<llvm::BasicBlock *, 32> SubCfgsBlocks;
  for (auto &SubCfg : SubCfgs)
    SubCfgsBlocks.insert(SubCfg.getNewBlocks().begin(), SubCfg.getNewBlocks().end());

  llvm::SmallVector<llvm::AllocaInst *, 8> WL;
  for (auto &I : *EntryBlock) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
      if (Alloca->hasMetadata(hipsycl::compiler::MDKind::Arrayified))
        continue; // already arrayified
      if (utils::anyOfUsers<llvm::Instruction>(
              Alloca, [&SubCfgsBlocks](llvm::Instruction *UI) { return !SubCfgsBlocks.contains(UI->getParent()); }))
        continue;
      if (!isAllocaSubCfgInternal(Alloca, SubCfgs, DT))
        WL.push_back(Alloca);
    }
  }

  for (auto *I : WL) {
    // todo: can we somehow enable this..?
    //    if (VecInfo.getVectorShape(*I).isUniform()) {
    //      HIPSYCL_DEBUG_INFO << "[SubCFG] Not arrayifying alloca " << *I << "\n";
    //      continue;
    //    }
    llvm::IRBuilder AllocaBuilder{I};
    llvm::Type *T = I->getAllocatedType();
    if (auto *ArrSizeC = llvm::dyn_cast<llvm::ConstantInt>(I->getArraySize())) {
      auto ArrSize = ArrSizeC->getLimitedValue();
      if (ArrSize > 1) {
        T = llvm::ArrayType::get(T, ArrSize);
        HIPSYCL_DEBUG_WARNING << "Caution, alloca was array\n";
      }
    }

    auto *Alloca = AllocaBuilder.CreateAlloca(T, AllocaBuilder.getInt32(ReqdArrayElements), I->getName() + "_alloca");
    Alloca->setAlignment(llvm::Align{hipsycl::compiler::DefaultAlignment});
    Alloca->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

    for (auto &SubCfg : SubCfgs) {
      auto *GepIp = SubCfg.getLoadBB()->getFirstNonPHIOrDbgOrLifetime();
      auto *ContiguousIdx = SubCfg.getHI().IsSub ? SubCfg.getHI().SGIdArg : SubCfg.getHI().ContiguousIdx;

      llvm::IRBuilder LoadBuilder{GepIp};
      auto *GEP = llvm::cast<llvm::GetElementPtrInst>(
          LoadBuilder.CreateInBoundsGEP(Alloca->getAllocatedType(), Alloca, ContiguousIdx, I->getName() + "_gep"));
      GEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

      llvm::replaceDominatedUsesWith(I, GEP, DT, SubCfg.getLoadBB());
    }
    I->eraseFromParent();
  }
}

void formSubCfgsGeneric(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
                        const SplitterAnnotationInfo &SAA, llvm::Loop *WILoop, std::size_t Dim,
                        const std::size_t ReqdArrayElements, llvm::ArrayRef<llvm::Value *> LocalSize,
                        HierarchicalSplitInfo HI) {
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(if (HI.IsSub) llvm::errs() << "[SubCFG] Transforming sub group\n";
                                else if (HI.HasSub) llvm::errs()
                                << "[SubCFG] Transforming work group containing sub barrier\n";
                                else llvm::errs() << "[SubCFG] Transforming work group\n"; F.viewCFG();)

  auto *Entry = &F.getEntryBlock();
  if (WILoop) {
    Entry = utils::getWorkItemLoopBodyEntry(WILoop);
  }

  llvm::DenseMap<llvm::BasicBlock *, size_t> Barriers;
  llvm::SmallVector<llvm::BasicBlock *, 4> ExitingBlocks;
  if (WILoop)
    ExitingBlocks.append(llvm::pred_begin(WILoop->getLoopLatch()), llvm::pred_end(WILoop->getLoopLatch()));
  else
    for (auto &BB : F)
      if (BB.getTerminator()->getNumSuccessors() == 0)
        ExitingBlocks.push_back(&BB);

  if (ExitingBlocks.empty()) {
    HIPSYCL_DEBUG_ERROR << "[SubCFG] Invalid kernel! No kernel exits!\n";
    llvm_unreachable("[SubCFG] Invalid kernel! No kernel exits!\n");
  }

  // mark exit barrier with the corresponding id:
  for (auto *BB : ExitingBlocks)
    Barriers[BB] = ExitBarrierId;
  // mark entry barrier with the corresponding id:
  Barriers[Entry] = EntryBarrierId;

  std::vector<llvm::BasicBlock *> Blocks;
  if (WILoop)
    Blocks.insert(Blocks.begin(), WILoop->block_begin(), WILoop->block_end());
  else {
    Blocks.reserve(std::distance(F.begin(), F.end()));
    std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });
  }

  // non-entry block Allocas are considered broken, move to entry.
  utils::moveAllocasToEntry(F, Blocks);
  // we need the load from __hipsycl_local_id_x, ..sg_id, .. to be unique.
  utils::moveGlobalVarLoadsToEntry(F, Blocks, LocalIdGlobalNameX);
  utils::moveGlobalVarLoadsToEntry(F, Blocks, LocalIdGlobalNameY);
  utils::moveGlobalVarLoadsToEntry(F, Blocks, LocalIdGlobalNameZ);
  utils::moveGlobalVarLoadsToEntry(F, Blocks, SgIdGlobalName);

  if (HI.IsSub) {
    HIPSYCL_DEBUG_INFO << "SGIDArg: " << *HI.SGIdArg << "\n";
    for (auto U : HI.SGIdArg->users()) {
      HIPSYCL_DEBUG_INFO << "SGIDArg user: " << *U << "\n";
    }
    HI.SGIdArg->replaceAllUsesWith(utils::getLoadForGlobalVariable(F, SgIdGlobalName));
    HI.SGIdArg = utils::getLoadForGlobalVariable(F, SgIdGlobalName);
  }

  auto RImpl = getRegion(F, LI, Blocks);
  hipsycl::compiler::Region R{*RImpl};
  auto VecInfo = getVectorizationInfo(F, R, LI, DT, PDT, Dim);
  if (!WILoop)
    VecInfo.setPinnedShape(*HI.ContiguousIdx, hipsycl::compiler::VectorShape::cont());

  // store all other barrier blocks with a unique id:
  for (auto *BB : Blocks)
    if (Barriers.find(BB) == Barriers.end() &&
        (HI.IsSub ? utils::hasOnlySubBarrier(BB, SAA) : utils::hasOnlyBarrier(BB, SAA)))
      Barriers.insert({BB, Barriers.size()});

  const llvm::DataLayout &DL = F.getParent()->getDataLayout();
  llvm::IRBuilder Builder{F.getEntryBlock().getFirstNonPHI()};
  auto *LastBarrierIdStorage =
      Builder.CreateAlloca(DL.getLargestLegalIntType(F.getContext()), nullptr, "LastBarrierId");

  // create subcfgs
  std::vector<SubCFG> SubCFGs;
  for (auto &BIt : Barriers) {
    HIPSYCL_DEBUG_INFO << "Create SubCFG from " << BIt.first->getName() << "(" << BIt.first << ") id: " << BIt.second
                       << "\n";
    if (BIt.second != ExitBarrierId)
      SubCFGs.emplace_back(BIt.first, LastBarrierIdStorage, Barriers, WILoop, SAA, Dim, HI);
  }

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> InstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> BaseInstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> InstContReplicaMap;

  for (auto &Cfg : SubCFGs)
    Cfg.arrayifyMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, SubCFGs,
                                  F.getEntryBlock().getFirstNonPHI(), ReqdArrayElements, VecInfo);

  llvm::BasicBlock *ExitFuncBB = nullptr;
  if (!WILoop)
    ExitFuncBB = ExitingBlocks[0];

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> RemappedInstAllocaMap;
  for (auto &Cfg : SubCFGs) {
    Cfg.print();
    if (WILoop)
      Cfg.replicate(WILoop, InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, RemappedInstAllocaMap);
    else
      Cfg.replicate(F, InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, RemappedInstAllocaMap, ExitFuncBB,
                    LocalSize, SAA);
    purgeLifetime(Cfg);
  }

  llvm::BasicBlock *WhileHeader = nullptr;
  if (WILoop)
    WhileHeader = generateWhileSwitchAround(WILoop->getLoopPreheader(), WILoop->getHeader(), WILoop->getExitBlock(),
                                            LastBarrierIdStorage, SubCFGs);
  else
    WhileHeader = generateWhileSwitchAround(&F.getEntryBlock(), F.getEntryBlock().getSingleSuccessor(), ExitFuncBB,
                                            LastBarrierIdStorage, SubCFGs);

  llvm::removeUnreachableBlocks(F);

  DT.recalculate(F);
  arrayifyAllocas(&F.getEntryBlock(), DT, SubCFGs, ReqdArrayElements, VecInfo);

  for (auto &Cfg : SubCFGs) {
    Cfg.fixSingleSubCfgValues(DT, RemappedInstAllocaMap, ReqdArrayElements, VecInfo);
    if (!WILoop && HI.HasSub) {
      auto Blocks = Cfg.getNewBlocks(); // copy
      Blocks.insert(Blocks.begin(), Cfg.getLoadBB());

      llvm::SetVector<llvm::Value *> Inputs, Outputs;
      llvm::CodeExtractorAnalysisCache CEAC{F};
      llvm::CodeExtractor CE{Blocks};
      assert(CE.isEligible());

      llvm::ValueToValueMapTy VMap;
#if LLVM_VERSION_MAJOR >= 14
      auto NewF = CE.extractCodeRegion(CEAC, Inputs, Outputs);

      HIPSYCL_DEBUG_INFO << "Inputs:"
                         << "\n";
      int Cnter = 0;
      for (auto I : Inputs) {
        HIPSYCL_DEBUG_INFO << *I << " -> " << *NewF->getArg(Cnter) << "\n";
        VMap[I] = NewF->getArg(Cnter++);
      }
      HIPSYCL_DEBUG_INFO << "Outputs:"
                         << "\n";
      for (auto O : Outputs) {
        HIPSYCL_DEBUG_INFO << *O << " -> " << *NewF->getArg(Cnter) << "\n";
        VMap[O] = NewF->getArg(Cnter++);
      }
#else
      // LLVM < 14 does not expose the In-/Outputs. should not really matter, since we fall back to global / undef loads
      // anyways, in case a required value is not used as input..
      auto NewF = CE.extractCodeRegion(CEAC);

      assert(NewF->hasOneUser());
      auto NewFCall = llvm::cast<llvm::CallBase>(NewF->user_back());
      auto OpIt = NewFCall->arg_begin();
      auto ArgIt = NewF->arg_begin();

      HIPSYCL_DEBUG_INFO << "In-/Outputs:"
                         << "\n";
      for (int i = 0; i < NewFCall->arg_size(); ++i) {
        HIPSYCL_DEBUG_INFO << **OpIt << " -> " << *ArgIt << "\n";
        VMap[*OpIt++] = ArgIt++;
      }
#endif

      utils::createSubBarrier(NewF->getEntryBlock().getTerminator(), const_cast<SplitterAnnotationInfo &>(SAA));
      for (auto &BB : *NewF)
        if (BB.getTerminator()->getNumSuccessors() == 0)
          utils::createSubBarrier(BB.getTerminator(), const_cast<SplitterAnnotationInfo &>(SAA));
      HIPSYCL_DEBUG_EXECUTE_VERBOSE(llvm::errs() << "extracted fn: \n" << *NewF; NewF->viewCFG();)

      llvm::DominatorTree NewDT{*NewF};
      llvm::PostDominatorTree NewPDT{*NewF};
      llvm::LoopInfo NewLI{NewDT};

      llvm::SmallVector<llvm::Value *, 3> NewIndVars;
      int D = 0;
      std::transform(Cfg.getWIIndVars().begin(), Cfg.getWIIndVars().end(), std::back_inserter(NewIndVars),
                     [&VMap, NewF, &D](llvm::Value *V) -> llvm::Value * {
                       D++;
                       HIPSYCL_DEBUG_INFO << "index mapping: " << *V << "\n";
                       if (auto I = llvm::dyn_cast<llvm::Instruction>(V); I && I->getParent()->getParent() == NewF) {
                         HIPSYCL_DEBUG_INFO << " meself --> " << *I << "\n";
                         return I;
                       }
                       if (auto It = VMap.find(V); It != VMap.end()) {
                         HIPSYCL_DEBUG_INFO << " --> " << *It->second << "\n";
                         return It->second;
                       }
                       HIPSYCL_DEBUG_INFO << "meh --> "
                                          << *utils::getLoadForGlobalVariable(*NewF, LocalIdGlobalNames[D - 1]) << "\n";
                       return utils::getLoadForGlobalVariable(*NewF, LocalIdGlobalNames[D - 1]);
                     });

      auto GetFromVMapOrLoad = [&VMap, NewF](llvm::Value *V) -> llvm::Value * {
        HIPSYCL_DEBUG_INFO << "get from vmap: " << *V << "\n";
        if (auto It = VMap.find(V); It != VMap.end()) {
          HIPSYCL_DEBUG_INFO << " --> VMapped : " << *It->second << "\n";
          return It->second;
        }
        HIPSYCL_DEBUG_INFO << " --> load from undef ptr: " << *V << "\n";
        return llvm::IRBuilder{NewF->getEntryBlock().getFirstNonPHI()}.CreateLoad(
            V->getType(), llvm::UndefValue::get(llvm::PointerType::get(V->getType(), 0)));
      };
      llvm::SmallVector<llvm::Value *, 3> NewLocalSize;
      std::array<std::string, 3> LocalSizeGlobalNames{"__hipsycl_local_size_x", "__hipsycl_local_size_y",
                                                      "__hipsycl_local_size_z"};
      D = 0;
      std::transform(LocalSize.begin(), LocalSize.end(), std::back_inserter(NewLocalSize),
                     [&VMap, NewF, &LocalSizeGlobalNames, &D](llvm::Value *V) -> llvm::Value * {
                       HIPSYCL_DEBUG_INFO << "get from vmap: " << *V << "\n";
                       ++D;
                       if (auto It = VMap.find(V); It != VMap.end()) {
                         HIPSYCL_DEBUG_INFO << " --> VMapped : " << *It->second << "\n";
                         return It->second;
                       }
                       HIPSYCL_DEBUG_INFO << " --> load from global: " << *V << "\n";
                       return utils::getLoadForGlobalVariable(*NewF, LocalSizeGlobalNames[D - 1], V->getType());
                     });
      llvm::Value *NewIndVar = GetFromVMapOrLoad(HI.ContiguousIdx);
      llvm::Value *SGIdArg = GetFromVMapOrLoad(utils::getLoadForGlobalVariable(F, SgIdGlobalName));

      llvm::ValueToValueMapTy GlobalVarToIdxMap;
      GlobalVarToIdxMap[NewIndVar] = HI.ContiguousIdx;
      for (size_t D = 0; D < LocalSize.size(); ++D) {
        GlobalVarToIdxMap[NewIndVars[D]] = Cfg.getWIIndVars()[D];
        HIPSYCL_DEBUG_INFO << "newlocal: " << *NewLocalSize[D] << " outer " << *LocalSize[D] << "\n";
        GlobalVarToIdxMap[NewLocalSize[D]] = LocalSize[D];
      }

      formSubCfgsGeneric(*NewF, NewLI, NewDT, NewPDT, SAA, nullptr, 1, SGSize,
                         {llvm::ConstantInt::get(LocalSize[0]->getType(), SGSize)},
                         {true, false, NewLocalSize, NewIndVars, NewIndVar, SGIdArg});

      {
        for (size_t D = 0; D < LocalSize.size(); ++D) {
          // GlobalVarToIdxMap[NewIndVars[D]] = Cfg.getWIIndVars()[D];
          HIPSYCL_DEBUG_INFO << "newlocal: " << *NewLocalSize[D] << " uses: " << NewLocalSize[D]->getNumUses()
                             << " outer " << *LocalSize[D] << "\n";
          GlobalVarToIdxMap[NewLocalSize[D]] = LocalSize[D];
        }
        llvm::SmallVector<llvm::BasicBlock *, 8> NewBlocks{};
        NewBlocks.reserve(std::distance(F.begin(), F.end()));
        std::transform(F.begin(), F.end(), std::back_inserter(NewBlocks), [](auto &BB) { return &BB; });
        llvm::remapInstructionsInBlocks(NewBlocks, GlobalVarToIdxMap);
        assert(SGIdArg->getNumUses() == 0);
      }

      assert(NewF->hasOneUser());
      utils::checkedInlineFunction(llvm::cast<llvm::CallBase>(NewF->user_back()), "[SubCFG]");
      assert(NewF->user_empty());
      NewF->eraseFromParent();
      llvm::SmallVector<llvm::BasicBlock *> FunBlocks;
      std::transform(F.begin(), F.end(), std::back_inserter(FunBlocks), [](llvm::BasicBlock &BB) { return &BB; });

      for (size_t D = 0; D < LocalSize.size(); ++D) {
        GlobalVarToIdxMap[utils::getLoadForGlobalVariable(F, LocalSizeGlobalNames[D])] = LocalSize[D];
      }
      llvm::remapInstructionsInBlocks(FunBlocks, GlobalVarToIdxMap);
      utils::moveGlobalVarLoadsToEntry(F, FunBlocks, SgIdGlobalName);

      for (auto &VarName : LocalSizeGlobalNames) {
        if (auto GV = F.getParent()->getGlobalVariable(VarName)) {
          llvm::SmallVector<llvm::LoadInst *> WL;
          for (auto *U : GV->users())
            if (auto LI = llvm::dyn_cast<llvm::LoadInst>(U); LI && LI->user_empty())
              WL.push_back(LI);
          for (auto *LI : WL)
            LI->eraseFromParent();
          assert(GV->user_empty());
          GV->eraseFromParent();
        }
      }
    }
  }

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  assert(!llvm::verifyFunction(F, &llvm::errs()) && "Function verification failed");

  // simplify while loop to get single latch that isn't marked as wi-loop to prevent misunderstandings.
  auto *WhileLoop = utils::updateDtAndLi(LI, DT, WhileHeader, F);
  HIPSYCL_DEBUG_INFO << "updated di and li\n";
  llvm::simplifyLoop(WhileLoop, &DT, &LI, nullptr, nullptr, nullptr, false);
  HIPSYCL_DEBUG_INFO << "simplified loops\n";
}

void formSubCfgs(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
                 const SplitterAnnotationInfo &SAA) {
  auto *WILoop = utils::getSingleWorkItemLoop(LI);
  //  assert(WILoop && "Must have work item loop in kernel");
  if (WILoop) {
    assert(WILoop->getCanonicalInductionVariable() && "Must have work item index");
  }

  const std::size_t Dim = utils::getRangeDim(F);
  HIPSYCL_DEBUG_INFO << "[SubCFG] Kernel is " << Dim << "-dimensional\n";

  const auto LocalSize = utils::getLocalSizeValues(F, Dim);

  const std::size_t ReqdArrayElements = utils::getReqdStackElements(F);

  // get a common (pseudo) index value to be replaced by the actual index later
  llvm::Instruction *IndVar = nullptr;
  if (WILoop) {
    IndVar = WILoop->getCanonicalInductionVariable();
  } else {
    llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
    auto *IndVarT = utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])->getType();
    IndVar = Builder.CreateLoad(IndVarT, llvm::UndefValue::get(llvm::PointerType::get(IndVarT, 0)), "contiguousIdx");
  }

  formSubCfgsGeneric(F, LI, DT, PDT, SAA, WILoop, Dim, ReqdArrayElements, LocalSize,
                     {false, utils::hasSubBarriers(F, SAA), {}, {}, IndVar, nullptr});

  if (!WILoop) {
    for (auto U : IndVar->users()) {
      HIPSYCL_DEBUG_ERROR << "IndVar still in use: " << *U << "\n";
    }
    IndVar->replaceAllUsesWith(llvm::UndefValue::get(IndVar->getType()));
    IndVar->eraseFromParent();
  }
}

void createLoopsAroundKernel(llvm::Function &F, llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                             llvm::PostDominatorTree &PDT) {
  auto *Body =
      llvm::SplitBlock(&F.getEntryBlock(), &*F.getEntryBlock().getFirstInsertionPt(), &DT, &LI, nullptr, "wibody"
#if LLVM_VERSION_MAJOR >= 13
                       ,
                       true
#endif
      );
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG());
#if LLVM_VERSION_MAJOR >= 13
  Body = Body->getSingleSuccessor();
#endif

  llvm::BasicBlock *ExitBB = nullptr;
  for (auto &BB : F) {
    if (BB.getTerminator()->getNumSuccessors() == 0) {
      ExitBB = llvm::SplitBlock(&BB, BB.getTerminator(), &DT, &LI, nullptr, "exit"
#if LLVM_VERSION_MAJOR >= 13
                                ,
                                true
#endif
      );
#if LLVM_VERSION_MAJOR >= 13
      if (Body == &BB)
        std::swap(Body, ExitBB);
      ExitBB = &BB;
#endif
      break;
    }
  }

  llvm::SmallVector<llvm::BasicBlock *, 8> Blocks{};
  Blocks.reserve(std::distance(F.begin(), F.end()));
  std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });

  utils::moveAllocasToEntry(F, Blocks);

  const auto Dim = utils::getRangeDim(F);

  llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
  auto *IndVarT = utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])->getType();
  llvm::Value *Idx = Builder.CreateLoad(IndVarT, llvm::UndefValue::get(llvm::PointerType::get(IndVarT, 0)));

  auto LocalSize = utils::getLocalSizeValues(F, Dim);
  llvm::ValueToValueMapTy VMap;
  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  auto *LastHeader = Body;

  createLoopsAround(F, ExitBB, LocalSize, 0, {false, false, {}, {}, Idx}, VMap, Latches, LastHeader, Idx);

  F.getEntryBlock().getTerminator()->setSuccessor(0, LastHeader);
  llvm::remapInstructionsInBlocks(Blocks, VMap);
  for (int D = 0; D < Dim; ++D)
    if (auto *Load = llvm::cast_or_null<llvm::LoadInst>(utils::getLoadForGlobalVariable(F, LocalIdGlobalNames[D])))
      Load->eraseFromParent();
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG())
}

} // namespace

namespace hipsycl::compiler {
void SubCfgFormationPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addRequiredTransitive<llvm::DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<llvm::PostDominatorTreeWrapperPass>();
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool SubCfgFormationPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  if (!SAA.isKernelFunc(&F))
    return false;

  HIPSYCL_DEBUG_INFO << "[SubCFG] Form SubCFGs in " << F.getName() << "\n";

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &PDT = getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();

  if (utils::hasBarriers(F, SAA) || utils::hasSubBarriers(F, SAA))
    formSubCfgs(F, LI, DT, PDT, SAA);
  else if (!utils::getSingleWorkItemLoop(LI))
    createLoopsAroundKernel(F, DT, LI, PDT);

  return false;
}

char SubCfgFormationPassLegacy::ID = 0;

llvm::PreservedAnalyses SubCfgFormationPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F))
    return llvm::PreservedAnalyses::all();

  HIPSYCL_DEBUG_INFO << "[SubCFG] Form SubCFGs in " << F.getName() << "\n";

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  if (utils::hasBarriers(F, *SAA) || utils::hasSubBarriers(F, *SAA))
    formSubCfgs(F, LI, DT, PDT, *SAA);
  else if (!utils::getSingleWorkItemLoop(LI))
    createLoopsAroundKernel(F, DT, LI, PDT);

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
