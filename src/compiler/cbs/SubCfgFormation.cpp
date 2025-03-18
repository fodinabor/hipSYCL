/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include "hipSYCL/compiler/cbs/SubCfgFormation.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/cbs/UniformityAnalysis.hpp"
#include "hipSYCL/compiler/cbs/CBSIntrinsics.hpp"

#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

#include "hipSYCL/common/debug.hpp"
#include <hipSYCL/RV.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/IVDescriptors.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/LoopPass.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/Regex.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/CodeExtractor.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>

#include <cstddef>
#include <functional>
#include <numeric>

#define DEBUG_SUBCFG_FORMATION

#define PASS_PREFIX_STR "[SubCFG]"

namespace hipsycl::compiler::cbs {
  struct State {
    size_t Dim;
    llvm::Type *SizeT;
    std::array<char, 3> DimName = {'x', 'y', 'z'};
    std::array<const char*, 3> LocalIdGlobalNames = LocalIdGlobalNames;
    std::array<const char*, 3> LocalSizeGlobalNames = LocalSizeGlobalNames;
  };

  llvm::LoadInst *mergeGVLoadsInEntry(llvm::Function &F, llvm::StringRef VarName,
                                      llvm::Type *ty) {
    auto SizeT = F.getParent()->getDataLayout().getLargestLegalIntType(F.getContext());
    auto *GV = F.getParent()->getOrInsertGlobal(VarName, ty ? ty : SizeT);

    llvm::LoadInst *FirstLoad = nullptr;
    llvm::SmallVector<llvm::LoadInst *, 4> Loads;
    for (auto U : GV->users()) {
      if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U); LI && LI->getFunction() == &F) {
        if (!FirstLoad)
          FirstLoad = LI;
        else
          Loads.push_back(LI);
      }
    }

    if (FirstLoad) {
      if (Loads.size() > 0 || FirstLoad->getParent() != &F.getEntryBlock()) {
        FirstLoad->moveBefore(&F.getEntryBlock().front());
        for (auto *LI : Loads) {
          LI->replaceAllUsesWith(FirstLoad);
          LI->eraseFromParent();
        }
      }
      return FirstLoad;
    }

    llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
    auto *Load = Builder.CreateLoad(ty ? ty : SizeT, GV, "cbs.load." + GV->getName());
    return Load;
  }
}

using namespace hipsycl::compiler;
using namespace hipsycl::compiler::cbs;

namespace {


bool isLoadFromGV(llvm::Value* V, llvm::Function &F, llvm::StringRef VarName) {
  if (!V) {
    return false;
  }
  auto SizeT = F.getParent()->getDataLayout().getLargestLegalIntType(F.getContext());
  auto *GV = F.getParent()->getOrInsertGlobal(VarName, SizeT);
  for (auto U : GV->users()) {
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U); LI && LI->getFunction() == &F) {
      if (V == LI) {
        return true;
      }
    }
  }
  return false;
}

// parses the range dimensionality from the mangled kernel name
std::size_t getRangeDim(llvm::Function &F) {
  auto FName = F.getName();
  // todo: fix with MS mangling
  llvm::Regex Rgx("iterate_nd_range_ompILi([1-3])E");
  llvm::SmallVector<llvm::StringRef, 4> Matches;
  if (Rgx.match(FName, &Matches))
    return std::stoull(static_cast<std::string>(Matches[1]));

  if (auto MD = F.getParent()->getNamedMetadata(SscpAnnotationsName)) {
    for (auto OP : MD->operands()) {
      if (OP->getNumOperands() == 3 &&
          llvm::cast<llvm::MDString>(OP->getOperand(1))->getString() == SscpKernelDimensionName) {
        if (&F == llvm::dyn_cast<llvm::Function>(
                      llvm::cast<llvm::ValueAsMetadata>(OP->getOperand(0))->getValue())) {
          auto ConstMD = llvm::cast<llvm::ConstantAsMetadata>(OP->getOperand(2))->getValue();
          if (auto CI = llvm::dyn_cast<llvm::ConstantInt>(ConstMD))
            return CI->getZExtValue();
          if (auto ZI = llvm::dyn_cast<llvm::ConstantAggregateZero>(ConstMD))
            return 0;
          if (auto CS = llvm::dyn_cast<llvm::ConstantStruct>(ConstMD))
            return llvm::cast<llvm::ConstantInt>(CS->getOperand(0))->getZExtValue();
        }
      }
    }
  }
  llvm_unreachable("[SubCFG] Could not deduce kernel dimensionality!");
}

// identify the local size values by the store to it
void fillStores(llvm::Value *V, int Idx, llvm::SmallVector<llvm::Value *, 3> &LocalSize) {
  if (const auto *Store = llvm::dyn_cast<llvm::StoreInst>(V)) {
    LocalSize[Idx] = Store->getOperand(0);
  } else if (auto *BC = llvm::dyn_cast<llvm::BitCastInst>(V)) {
    for (auto *BCU : BC->users()) {
      fillStores(BCU, Idx, LocalSize);
    }
  } else if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(V)) {
    auto *IdxV = GEP->indices().begin() + (GEP->getNumIndices() - 1);
    const auto *IdxC = llvm::cast<llvm::ConstantInt>(IdxV);
    for (auto *GU : GEP->users()) {
      fillStores(GU, IdxC->getSExtValue(), LocalSize);
    }
  }
}

// reinterpret single argument as array if neccessary and load scalar size values into LocalSize
void loadSizeValuesFromArgument(llvm::Function &F, llvm::Value *LocalSizeArg,
                                const llvm::DataLayout &DL,
                                llvm::SmallVector<llvm::Value *, 3> &LocalSize, State state) {
  // local_size is just an array of size_t's..
  auto SizeTSize = DL.getLargestLegalIntTypeSizeInBits();
  auto *SizeT = DL.getLargestLegalIntType(F.getContext());

  llvm::IRBuilder Builder{F.getEntryBlock().getFirstNonPHI()};
  llvm::Value *LocalSizePtr = nullptr;
  if (!LocalSizeArg->getType()->isArrayTy()) {
#if HAS_TYPED_PTR
    auto PtrTy = llvm::Type::getIntNPtrTy(F.getContext(), SizeTSize);
#else
    auto PtrTy = llvm::PointerType::get(F.getContext(), 0);
#endif
    LocalSizePtr = Builder.CreatePointerCast(LocalSizeArg, PtrTy, "local_size.cast");
  }
  for (unsigned int I = 0; I < state.Dim; ++I) {
    auto CurDimName = state.DimName[I];
    if (LocalSizeArg->getType()->isArrayTy()) {
      LocalSize[I] =
          Builder.CreateExtractValue(LocalSizeArg, {I}, "local_size." + llvm::Twine{CurDimName});
    } else {
      auto *LocalSizeGep =
          Builder.CreateInBoundsGEP(SizeT, LocalSizePtr, Builder.getIntN(SizeTSize, I),
                                    "local_size.gep." + llvm::Twine{CurDimName});
      HIPSYCL_DEBUG_INFO << *LocalSizeGep << "\n";

      LocalSize[I] =
          Builder.CreateLoad(SizeT, LocalSizeGep, "local_size." + llvm::Twine{CurDimName});
    }
  }
}

// get the wg size values for the loop bounds
llvm::SmallVector<llvm::Value *, 3> getLocalSizeValues(llvm::Function &F, const State &state) {
  llvm::SmallVector<llvm::Value *, 3> LocalSize(state.Dim);
  for (int I = 0; I < state.Dim; ++I) {
    auto Load = mergeGVLoadsInEntry(F, state.LocalSizeGlobalNames[I]);
    Load->moveBefore(F.getEntryBlock().getTerminator());
    LocalSize[I] = Load;
  }
  return LocalSize;
}

llvm::SmallVector<llvm::Value *, 3> loadLocalSizesFromAnnotations(llvm::Function &F, State state) {
  auto &DL = F.getParent()->getDataLayout();
  auto [LocalSizeArg, Annotation] = utils::getLocalSizeArgumentFromAnnotation(F);

  llvm::SmallVector<llvm::Value *, 3> LocalSize(state.Dim);
  HIPSYCL_DEBUG_INFO << *LocalSizeArg << "\n";

  if (!llvm::dyn_cast<llvm::Argument>(LocalSizeArg))
    for (auto *U : LocalSizeArg->users())
      fillStores(U, 0, LocalSize);
  else
    loadSizeValuesFromArgument(F, LocalSizeArg, DL, LocalSize, state);

  Annotation->eraseFromParent();
  return LocalSize;
}

std::unique_ptr<RegionImpl> getRegion(llvm::Function &F, const llvm::LoopInfo &LI,
                                      llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  return std::unique_ptr<RegionImpl>{new FunctionRegion(F, Blocks)};
}

// calculate uniformity analysis
VectorizationInfo getVectorizationInfo(llvm::Function &F, Region &R, llvm::LoopInfo &LI,
                                       llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
                                       size_t Dim, State state, HierarchicalSplitInfo HI) {
  VectorizationInfo VecInfo{F, R};
  // seed varyingness
  for (size_t D = 0; D < Dim - 1; ++D) {
    VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, state.LocalIdGlobalNames[D]),
                           VectorShape::cont());
  }
  VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, state.LocalIdGlobalNames[Dim - 1]),
                         VectorShape::cont());
  VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, cbs::SgLocalIdGlobalName), VectorShape::cont());
  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, cbs::SgIdGlobalName), VectorShape::uni());
  }
  if (HI.Level == HierarchicalLevel::CBS || HI.Level == HierarchicalLevel::H_CBS_GROUP) {
    VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, cbs::SgIdGlobalName), VectorShape::cont());
  }
  VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, cbs::SgNumSubgroupsGlobalName), VectorShape::uni());
  VecInfo.setPinnedShape(*HI.ContiguousIdx, VectorShape::cont());

  auto* VoidPointerType = llvm::PointerType::get(F.getContext(), F.getParent()->getDataLayout().getAllocaAddrSpace());
  VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, cbs::SubGroupSharedMemory, VoidPointerType), VectorShape::uni());
  VecInfo.setPinnedShape(*mergeGVLoadsInEntry(F, cbs::WorkGroupSharedMemory, VoidPointerType), VectorShape::uni());

  VectorizationAnalysis VecAna{VecInfo, LI, DT, PDT};
  VecAna.analyze();
  return VecInfo;
}

// create the wi-loops around a kernel or subCFG, LastHeader input should be the load block,
// ContiguousIdx may be any identifyable value (load from undef)
void createLoopsAround(llvm::Function &F, llvm::BasicBlock *AfterBB,
                       const llvm::ArrayRef<llvm::Value *> &LocalSize, int EntryId,
                       llvm::ValueToValueMapTy &VMap,
                       llvm::SmallVector<llvm::BasicBlock *, 3> &Latches,
                       llvm::BasicBlock *&LastHeader, llvm::Value *&ContiguousIdx, State state,
                       HierarchicalSplitInfo& HI, llvm::SmallVector<llvm::PHINode *, 3>& IndVars) {
  const auto &DL = F.getParent()->getDataLayout();
  auto *LoadBB = LastHeader;
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};

  const int Dim = HI.Level != HierarchicalLevel::H_CBS_SUBGROUP ? static_cast<int>(state.Dim) : 1;
  const auto InnerMost = Dim - 1;
  constexpr auto OuterMost = 0;

  auto loopIncrement = [&HI, &InnerMost](const int D) -> size_t {
    if (D != InnerMost) {
      return 1;
    }
    switch (HI.Level) {
    case HierarchicalLevel::CBS:
    case HierarchicalLevel::H_CBS_SUBGROUP:
      return 1;
    case HierarchicalLevel::H_CBS_GROUP:
      return SGSize;
    }
    assert(false);
    return 0;
  };

  auto suffix = [&HI, &state, &EntryId](int D) {
    return (llvm::Twine{HI.Level == HierarchicalLevel::H_CBS_SUBGROUP ? 's'
                                                                      : state.DimName[D]} +
            ".subcfg." + llvm::Twine{EntryId})
        .str();
  };

  // from innermost to outermost: create loops around the LastHeader and use AfterBB as dummy exit
  // to be replaced by the outer latch later
  for (int D = Dim - 1; D >= 0; --D) {
    const std::string Suffix = suffix(D);

    auto *Header = llvm::BasicBlock::Create(LastHeader->getContext(), "header." + Suffix + "b",
                                            LastHeader->getParent(), LastHeader);
    Builder.SetInsertPoint(Header, Header->getFirstInsertionPt());
    auto *WIIndVar =
        Builder.CreatePHI(DL.getLargestLegalIntType(F.getContext()), 2, "indvar." + Suffix);
    WIIndVar->addIncoming(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), 0),
                          &F.getEntryBlock());

    IndVars.push_back(WIIndVar);
    Builder.CreateBr(LastHeader);

    auto *Latch = llvm::BasicBlock::Create(F.getContext(), "latch." + Suffix + "b", &F);
    Builder.SetInsertPoint(Latch, Latch->getFirstInsertionPt());

    auto *IncIndVar = Builder.CreateAdd(
        WIIndVar, Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), loopIncrement(D)),
        "addInd." + Suffix, true, false);
    WIIndVar->addIncoming(IncIndVar, Latch);

    llvm::Value *LoopCond = Builder.CreateICmpULT(IncIndVar, LocalSize[D], "exit.cond." + Suffix);

#if not USE_RV
    if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
      assert(D == InnerMost);
      // Here, we need to use IncIndVar because we are in the loop latch and want
      // to check whether to execute a next iteration (and not check for the current iteration).
      // Condition: if the innermost dimension is not divisble by the sub-group size, then
      // the last sub-group in the dimension is incomplete.
      // This makes handling the incrementation of induction variables easier.
      // Since, we don't need to change the outer induction vars inside of sub-CFG.
      HI.InnerPhiInd = mergeGVLoadsInEntry(F, "__inner_ind_var");
      auto *ContCond =
          Builder.CreateICmpULT(Builder.CreateAdd(IncIndVar, HI.InnerPhiInd),
                                HI.InnerSize, "exit.cont_cond." + Suffix);
#if INCOMPLETE_SGS_OPT
      auto *noIncompleteSgs = mergeGVLoadsInEntry(F, "no-incomplete-sgs", ContCond->getType());
      ContCond = Builder.CreateLogicalOr(noIncompleteSgs, ContCond);
#endif

      LoopCond = Builder.CreateLogicalAnd(ContCond, LoopCond);
    }
#endif

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
  VMap[AfterBB] = Latches[InnerMost];

  // Mark work-item loop as parallel
  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP || HI.Level == HierarchicalLevel::CBS) {
    auto *MDWorkItemLoop = llvm::MDNode::get(
        F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::WorkItemLoop)});
    auto *LoopID =
        llvm::makePostTransformationMetadata(F.getContext(), nullptr, {}, {MDWorkItemLoop});
    Latches[InnerMost]->getTerminator()->setMetadata("llvm.loop", LoopID);
  }

  // add contiguous ind var calculation to load block
  Builder.SetInsertPoint(IndVars[InnerMost]->getParent(), ++IndVars[InnerMost]->getIterator());
  llvm::Value *Idx = IndVars[OuterMost];
  llvm::SmallVector<llvm::Value*, 3> Idxes{};
  Idxes.emplace_back(Idx);
  for (size_t D = 1; D < Dim; ++D) {
    const std::string Suffix =
        (llvm::Twine{state.DimName[D]} + ".subcfg." + llvm::Twine{EntryId}).str();

    Idx = Builder.CreateMul(Idx, LocalSize[D], "idx.mul." + Suffix, true);
    Idx = Builder.CreateAdd(IndVars[D], Idx, "idx.add." + Suffix, true);
    Idxes.emplace_back(Idx);
  }

  for (size_t D = 0; D < Dim; ++D) {
    VMap[mergeGVLoadsInEntry(F, state.LocalIdGlobalNames[D])] = IndVars[D];
  }


  Builder.SetInsertPoint(LoadBB, LoadBB->getFirstInsertionPt());
  if (HI.Level == HierarchicalLevel::CBS) {
    VMap[mergeGVLoadsInEntry(F, cbs::SgIdGlobalName)] = Builder.CreateUDiv(Idx, Builder.getInt64(SGSize));
    VMap[mergeGVLoadsInEntry(F, cbs::SgLocalIdGlobalName)] = Builder.CreateURem(IndVars.back(), llvm::ConstantInt::get(IndVars.back()->getType(), SGSize));
  } else if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    VMap[mergeGVLoadsInEntry(F, cbs::SgLocalIdGlobalName)] = Idx;
  } else {
    assert(HI.Level == HierarchicalLevel::H_CBS_GROUP);

    // InnerMost induction variable + SgSize
    VMap[mergeGVLoadsInEntry(F, state.LocalIdGlobalNames[InnerMost])] =
        Builder.CreateAdd( IndVars[InnerMost], mergeGVLoadsInEntry(F, cbs::SgLocalIdGlobalName));
    Idx = Builder.CreateAdd(Idx, mergeGVLoadsInEntry(F, cbs::SgLocalIdGlobalName));

    VMap[mergeGVLoadsInEntry(F, cbs::SgIdGlobalName)] = [&]() {
      auto InnerSgId = Builder.CreateUDiv(
    IndVars[InnerMost],
        Builder.getInt64(SGSize));

      if (Dim > 1) {
        // We round up
        auto *NumSgsInnerMostDimension = Builder.CreateUDiv(
            Builder.CreateAdd(LocalSize[InnerMost],
                              Builder.getIntN(state.SizeT->getIntegerBitWidth(), SGSize - 1)),
            Builder.getIntN(state.SizeT->getIntegerBitWidth(), SGSize));
        InnerSgId = Builder.CreateAdd(Builder.CreateMul(Idxes[InnerMost-1], NumSgsInnerMostDimension), InnerSgId);
      }
      return InnerSgId;
    }();

    auto *IterationsLeft = [&]() {
      auto *InnerDimIterationsLeft = Builder.CreateSub(LocalSize[InnerMost], IndVars[InnerMost]);
      auto *EnoughIterationsLeft = Builder.CreateICmpULE(Builder.getInt64(SGSize), InnerDimIterationsLeft);
#if INCOMPLETE_SGS_OPT
      auto *noIncompleteSgs = mergeGVLoadsInEntry(F, "no-incomplete-sgs", EnoughIterationsLeft->getType());
      EnoughIterationsLeft = Builder.CreateLogicalOr(noIncompleteSgs, EnoughIterationsLeft);
#endif
      return Builder.CreateSelect(EnoughIterationsLeft, Builder.getInt64(SGSize), InnerDimIterationsLeft);
    }();
    VMap[mergeGVLoadsInEntry(F, cbs::SgSizeGlobalName)] = IterationsLeft;
  }

  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    llvm::ValueToValueMapTy VMap;
    VMap[HI.ContiguousIdx] = Idx;
    VMap[mergeGVLoadsInEntry(F, state.LocalIdGlobalNames[InnerMost])] = Idx;
    llvm::SmallVector<llvm::BasicBlock *> Blocks{Latches.begin(), Latches.end()};
    llvm::remapInstructionsInBlocks(Blocks, VMap);
  }

  // in case code references all dimensions, we need to set the remaining dimensions to 0
  for (size_t D = Dim; D < 3; ++D) {
    auto ID = mergeGVLoadsInEntry(F, state.LocalIdGlobalNames[D]);
    ID->replaceAllUsesWith(Builder.getIntN(Idx->getType()->getIntegerBitWidth(), 0));
    ID->eraseFromParent();
  }

  VMap[ContiguousIdx] = Idx;
  ContiguousIdx = Idx;
}

void addRemappedDenseMapKeys(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &OrgInstAllocaMap,
    const llvm::ValueToValueMapTy &VMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &NewInstAllocaMap) {
  for (auto &[Inst, Alloca] : OrgInstAllocaMap) {
    if (auto *NewInst = llvm::dyn_cast_or_null<llvm::Instruction>(VMap.lookup(Inst)))
      NewInstAllocaMap.insert({NewInst, Alloca});
  }
}

// check if a contiguous value can be tracked back to only uniform values and the wi-loop indvar
// currently cannot track back the value through PHI nodes.
bool dontArrayifyValues(
    llvm::Instruction &I,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::Instruction *AllocaIP, llvm::Value *ReqdArrayElements, llvm::Value *IndVar,
    VectorizationInfo &VecInfo) {
  // is cont indvar
  if (VecInfo.isPinned(I))
    return true;

  auto &F = *I.getParent()->getParent();

  llvm::SmallVector<llvm::Instruction *, 4> WL;
  llvm::SmallPtrSet<llvm::Instruction *, 8> UniformValues;
  llvm::SmallVector<llvm::Instruction *, 8> ContiguousInsts;
  llvm::SmallPtrSet<llvm::Value *, 8> LookedAt;
  HIPSYCL_DEBUG_INFO << "[SubCFG] Cont value? " << I << " IndVar: " << *IndVar << "\n";
  WL.push_back(&I);
  while (!WL.empty()) {
    auto *WLValue = WL.pop_back_val();
    if (auto *WLI = llvm::dyn_cast<llvm::Instruction>(WLValue))
      for (auto *V : WLI->operand_values()) {
        HIPSYCL_DEBUG_INFO << "[SubCFG] Considering: " << *V << "\n";

        if (V == IndVar || VecInfo.isPinned(*V) || llvm::isa<llvm::Constant>(V))
          continue;
        // todo: fix PHIs
        if (LookedAt.contains(V))
          return false;
        LookedAt.insert(V);

        // collect cont and uniform source values
        if (auto *OpI = llvm::dyn_cast<llvm::Instruction>(V)) {
          if (!VecInfo.getVectorShape(*OpI).isUniform() ||
              isLoadFromGV(OpI, F, cbs::SgIdGlobalName) ||
              isLoadFromGV(OpI, F, cbs::SgLocalIdGlobalName)) {
            WL.push_back(OpI);
            ContiguousInsts.push_back(OpI);
          } else if (!UniformValues.contains(OpI))
            UniformValues.insert(OpI);
        } else {
          return false;
        }
      }
  }

  for (auto *UI : UniformValues) {
    HIPSYCL_DEBUG_INFO << "[SubCFG] UniValue to store: " << *UI << "\n";
    if (BaseInstAllocaMap.lookup(UI))
      continue;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Store required uniform value to single element alloca " << I
                       << "\n";
    auto *Alloca =
        utils::arrayifyInstruction(F.getEntryBlock().getFirstNonPHI(), UI, IndVar, nullptr);
    BaseInstAllocaMap.insert({UI, Alloca});
    VecInfo.setVectorShape(*Alloca, hipsycl::compiler::VectorShape::uni());
  }
  ContInstReplicaMap.insert({&I, ContiguousInsts});
  return true;
}


void remapInstruction(llvm::Instruction *I, llvm::ValueToValueMapTy &VMap) {
  llvm::SmallVector<llvm::Value *, 8> WL{I->value_op_begin(), I->value_op_end()};
  for (auto *V : WL) {
    if (VMap.count(V))
      I->replaceUsesOfWith(V, VMap[V]);
  }
  HIPSYCL_DEBUG_INFO << "[SubCFG] remapped Inst " << *I << "\n";
}

} // namespace

namespace hipsycl::compiler::cbs {

// create new exiting block writing the exit's id to LastBarrierIdStorage_
llvm::BasicBlock *
SubCFG::createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
                         llvm::BasicBlock *After, llvm::BasicBlock *TargetBB) {
  HIPSYCL_DEBUG_INFO << "Create new exit with ID: " << BarrierPair.second << " at "
                     << After->getName() << "\n";

  auto *Exit = llvm::BasicBlock::Create(After->getContext(),
                                        After->getName() + ".subcfg.exit" +
                                            llvm::Twine{BarrierPair.second} + "b",
                                        After->getParent(), TargetBB);

  auto &DL = Exit->getParent()->getParent()->getDataLayout();
  llvm::IRBuilder Builder{Exit, Exit->getFirstInsertionPt()};
  auto& F = *After->getParent()->getParent();
  auto *MDAlloca = llvm::MDNode::get(
      F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::LoopState)});
  auto Store = Builder.CreateStore(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), BarrierPair.second),
                      LastBarrierIdStorage_);
  Store->setMetadata(MDKind::Arrayified, MDAlloca);
  Builder.CreateBr(TargetBB);

  After->getTerminator()->replaceSuccessorWith(BarrierPair.first, Exit);
  return Exit;
}

// identify a new SubCFG using DFS starting at EntryBarrier
SubCFG::SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
               const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds,
               const SplitterAnnotationInfo &SAA, size_t Dim, HierarchicalSplitInfo HI)
    : EntryId_(BarrierIds.lookup(EntryBarrier)), EntryBarrier_(EntryBarrier),
      LastBarrierIdStorage_(LastBarrierIdStorage), EntryBB_(EntryBarrier->getSingleSuccessor()),
      LoadBB_(nullptr), PreHeader_(nullptr), Dim(Dim), HI(HI) {
  // if (!EntryBB_)
  //  EntryBarrier->getParent()->viewCFG();
  assert(EntryBB_);

  //assert(HI.ContiguousIdx && "Must have found __acpp_cbs_local_id_{x,y,z}");

  llvm::SmallVector<llvm::BasicBlock *, 4> WL{EntryBarrier};
  while (!WL.empty()) {
    auto *BB = WL.pop_back_val();

    llvm::SmallVector<llvm::BasicBlock *, 2> Succs{llvm::succ_begin(BB), llvm::succ_end(BB)};
    for (auto *Succ : Succs) {
      if (std::find(Blocks_.begin(), Blocks_.end(), Succ) != Blocks_.end())
        continue;

      if (!(HI.Level == HierarchicalLevel::H_CBS_SUBGROUP ? utils::hasOnlySubBarrier(Succ, SAA)
                                                          : utils::hasOnlyBarrier(Succ, SAA))) {
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
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *BB
                                  : Blocks_) {
    llvm::outs() << BB->getName() << ", ";
  } llvm::outs() << "\n";)
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

// clone all BBs of the subcfg, create wi-loop structure around and fixup values
void SubCFG::replicate(
    llvm::Function &F, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
    llvm::BasicBlock *AfterBB, llvm::ArrayRef<llvm::Value *> LocalSize, State state,
    llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca) {
  llvm::ValueToValueMapTy VMap;

  // clone blocks
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

  createLoopsAround(F, AfterBB, LocalSize, EntryId_, VMap, Latches, LastHeader, Idx, state, HI, WIPhiIndVars_);

  PreHeader_ = createUniformLoadBB(LastHeader);
  LastHeader->replacePhiUsesWith(&F.getEntryBlock(), PreHeader_);

  print();

  // TODO what is the purpose of RemappedInstAllocaMap, and why does it get filled before load...
  addRemappedDenseMapKeys(InstAllocaMap, VMap, RemappedInstAllocaMap);
  loadMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap,
                        loadToAlloca);
  loadUniformAndRecalcContValues(BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap,
                                 loadToAlloca, state);

  llvm::SmallVector<llvm::BasicBlock *, 8> BlocksToRemap{NewBlocks_.begin(), NewBlocks_.end()};
  llvm::remapInstructionsInBlocks(BlocksToRemap, VMap);

  removeDeadPhiBlocks(BlocksToRemap);

  EntryBB_ = PreHeader_;
  ExitBB_ = Latches[0];
  WILoopLatch = Latches.back();
  HI.ContiguousIdx = Idx;
}

// remove incoming PHI blocks that no longer actually have an edge to the PHI
void SubCFG::removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const {
  for (auto *BB : BlocksToRemap) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 4> Predecessors{llvm::pred_begin(BB), llvm::pred_end(BB)};
    for (auto &I : *BB) {
      if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&I)) {
        llvm::SmallVector<llvm::BasicBlock *, 4> IncomingBlocksToRemove;
        for (size_t IncomingIdx = 0; IncomingIdx < Phi->getNumIncomingValues(); ++IncomingIdx) {
          auto *IncomingBB = Phi->getIncomingBlock(IncomingIdx);
          if (!Predecessors.contains(IncomingBB))
            IncomingBlocksToRemove.push_back(IncomingBB);
        }
        for (auto *IncomingBB : IncomingBlocksToRemove) {
          HIPSYCL_DEBUG_INFO << "[SubCFG] Remove incoming block " << IncomingBB->getName()
                             << " from PHI " << *Phi << "\n";
          Phi->removeIncomingValue(IncomingBB);
          HIPSYCL_DEBUG_INFO << "[SubCFG] Removed incoming block " << IncomingBB->getName()
                             << " from PHI " << *Phi << "\n";
        }
      }
    }
  }
}

// creates array allocas for values that are identified as spanning multiple subcfgs
void SubCFG::arrayifyMultiSubCfgValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, llvm::Value *ReqdArrayElements,
    VectorizationInfo &VecInfo, llvm::Function &F) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> OtherCFGBlocks;
  for (auto &Cfg : SubCFGs) {
    if (&Cfg != this)
      OtherCFGBlocks.insert(Cfg.Blocks_.begin(), Cfg.Blocks_.end());
  }

  auto *ContiguousIdx = HI.ContiguousIdx;

  HIPSYCL_DEBUG_ERROR << "[SubCFG] ARRAIFY \n";

  for (auto *BB : Blocks_) {
    for (auto &I : *BB) {
      if (&I == ContiguousIdx || isLoadFromGV(&I, F, cbs::WorkGroupSharedMemory) ||
          isLoadFromGV(&I, F, cbs::SubGroupSharedMemory) ||
          isLoadFromGV(&I, F, cbs::SgIdGlobalName))
        continue;
      if (InstAllocaMap.lookup(&I)) {
        continue;
      }
      // if any use is in another subcfg
      if (utils::anyOfUsers<llvm::Instruction>(&I, [&OtherCFGBlocks, &I](auto *UI) {
            return (UI->getParent() != I.getParent() ||
                    UI->getParent() == I.getParent() && UI->comesBefore(&I)) &&
                   OtherCFGBlocks.contains(UI->getParent());
          })) {
        HIPSYCL_DEBUG_ERROR << "[SubCFG] USE in another subcfg \n";
        // load from an alloca, just widen alloca
        if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(&I))
          if (auto *Alloca = utils::getLoopStateAllocaForLoad(*LInst)) {
            InstAllocaMap.insert({&I, Alloca});
            continue;
          }

        HIPSYCL_DEBUG_ERROR << "arrayifyMultiSubCfgValues: " << I << "\n";
        // GEP from already widened alloca: reuse alloca
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
          if (GEP->hasMetadata(MDKind::Arrayified)) {
            // If we are on the subgroup level, then the gep operand might be an argument
            // and not directly an alloca.
            auto *GepPointerOperand = GEP->getPointerOperand();
            if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
              if (auto *Argument = llvm::dyn_cast<llvm::Argument>(GepPointerOperand)) {
                GepPointerOperand = (*HI.ArgsToAloca)[Argument];
              }
            }
            InstAllocaMap.insert({&I, llvm::cast<llvm::AllocaInst>(GepPointerOperand)});
            continue;
          }

        auto Shape = VecInfo.getVectorShape(I);
        HIPSYCL_DEBUG_ERROR << "VECTOR INFO: " << Shape << "\n";

        const auto isTrivialStepAway = [&F](llvm::Instruction &I,  llvm::StringRef S) {
          auto getInsideRvUniform = [](llvm::Value* V)-> llvm::Value* {
            if (V == nullptr) {
              return nullptr;
            }
            if (auto *OpI = llvm::dyn_cast<llvm::Instruction>(V)) {
              if (const auto CallInst = llvm::dyn_cast<llvm::CallInst>(OpI)) {
                if (CallInst->getCalledFunction()->getName().contains("rv_is_uniform")) {
                  return CallInst->getOperand(0);
                }
              }
            }
            return nullptr;
          };
          auto *V = [&]() -> llvm::Value* {
            if (I.isBinaryOp()) {
              if (llvm::dyn_cast<llvm::Constant>(I.getOperand(0))) {
                return I.getOperand(1);
              }
              return I.getOperand(0);
            }
            if (I.getOpcode() == llvm::Instruction::Trunc or
                I.getOpcode() == llvm::Instruction::ZExt or
                I.getOpcode() == llvm::Instruction::SExt or
                I.getOpcode() == llvm::Instruction::BitCast) {
              return I.getOperand(0);
                }
            return nullptr;
          }();

          return isLoadFromGV(&I, F, S) || isLoadFromGV(getInsideRvUniform(&I), F, S) || isLoadFromGV(V, F, S) || isLoadFromGV(getInsideRvUniform(V), F, S);
        };

        const bool UsedByCbsIntrinsic = utils::anyOfUsers<llvm::Instruction>(&I, [](auto *UI) {
          if (auto *CallInst = llvm::dyn_cast<llvm::CallInst>(UI);
              CallInst and CallInst->getCalledFunction()) {
            return CallInst->getCalledFunction()->getName().contains("__cbs_");
          }
          return false;
        });

        // if contiguous, and can be recalculated, don't arrayify but store
        // uniform values and insts required for recalculation
        if (!UsedByCbsIntrinsic && (isTrivialStepAway(I, cbs::SgIdGlobalName))) {
          if (dontArrayifyValues(I, BaseInstAllocaMap, ContInstReplicaMap, AllocaIP,
                                 ReqdArrayElements, ContiguousIdx, VecInfo)) {
            HIPSYCL_DEBUG_INFO << "[SubCFG] Not arrayifying " << I << "\n";
            continue;
          }
          HIPSYCL_DEBUG_INFO << "[SubCFG] considered not arrayifing but decided not to " << I << "\n";
        }

#ifndef HIPSYCL_NO_PHIS_IN_SPLIT
        // if value is uniform, just store to 1-wide alloca
        if (Shape.isUniform()) {
          HIPSYCL_DEBUG_INFO << "[SubCFG] Value uniform, store to single element alloca " << I
                             << "\n";
          auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, ContiguousIdx, nullptr);
          InstAllocaMap.insert({&I, Alloca});
          VecInfo.setVectorShape(*Alloca, VectorShape::uni());
          continue;
        }
#endif

        // create wide alloca and store the value
        auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, ContiguousIdx, ReqdArrayElements);
        InstAllocaMap.insert({&I, Alloca});
        VecInfo.setVectorShape(*Alloca, Shape);
      }
    }
  }
}

// inserts loads from the loop state allocas for varying values that were identified as
// multi-subcfg values
void SubCFG::loadMultiSubCfgValues(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap,
    llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &LoadToAlloca) {
  llvm::Value *NewContIdx = VMap[HI.ContiguousIdx];

  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  llvm::IRBuilder Builder{LoadTerm};

  for (auto &[Inst, Alloca] : InstAllocaMap) {
    // If def not in sub CFG but a use of it is in the sub CFG
    if (std::find(Blocks_.begin(), Blocks_.end(), Inst->getParent()) == Blocks_.end()) {
      if (utils::anyOfUsers<llvm::Instruction>(Inst, [this](llvm::Instruction *UI) {
            return std::find(NewBlocks_.begin(), NewBlocks_.end(), UI->getParent()) !=
                   NewBlocks_.end();
          })) {
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Inst)) {
          if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
            llvm::Value *ContIdx = VMap[HI.ContiguousIdx];
            auto *Type = GEP->getSourceElementType();
            auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(Builder.CreateInBoundsGEP(
                Type, GEP->getPointerOperand(), ContIdx, GEP->getName() + "c"));
            NewGEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDArrayified);
            VMap[Inst] = NewGEP;
            continue;
          }
        }
        auto *IP = LoadTerm;
        if (!Alloca->isArrayAllocation())
          IP = UniformLoadTerm;
        HIPSYCL_DEBUG_ERROR << "[SubCFG] Load from Alloca " << *Alloca << " in "
                            << IP->getParent()->getName() << "\n";
        auto *Load = utils::loadFromAlloca(Alloca, NewContIdx, IP, Inst->getName());
        LoadToAlloca[Load] = Alloca;

        utils::copyDgbValues(Inst, Load, IP);
        VMap[Inst] = Load;
      }
    }
  }
}

// Inserts loads for the multi-subcfg values that were identified as uniform
// inside the wi-loop preheader. Additionally clones the instructions that were
// identified as contiguous \a ContInstReplicaMap inside the LoadBB_ to restore
// the contiguous value just from the uniform values and the wi-idx.
void SubCFG::loadUniformAndRecalcContValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
        &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap,
    llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &LoadToAlloca, State state) {

  llvm::ValueToValueMapTy UniVMap;
  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  auto ContiguousIdx = HI.ContiguousIdx;
  llvm::Value *NewContIdx = VMap[ContiguousIdx];
  UniVMap[ContiguousIdx] = NewContIdx;
  auto& F = *this->LoadBB_->getParent();
  if (HI.Level == HierarchicalLevel::CBS || HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    UniVMap[mergeGVLoadsInEntry(F, cbs::SgLocalIdGlobalName)] = VMap[mergeGVLoadsInEntry(F, cbs::SgLocalIdGlobalName)];

  }
  if (HI.Level == HierarchicalLevel::CBS || HI.Level == HierarchicalLevel::H_CBS_GROUP) {
    UniVMap[mergeGVLoadsInEntry(F, cbs::SgIdGlobalName)] = VMap[mergeGVLoadsInEntry(F, cbs::SgIdGlobalName)];
  }

  // copy local id load value to univmap
  for (size_t D = 0; D < this->Dim; ++D) {
    auto *Load =
        mergeGVLoadsInEntry(*this->LoadBB_->getParent(), state.LocalIdGlobalNames[D]);
    UniVMap[Load] = VMap[Load];
  }

  // load uniform values from allocas
  for (auto &InstAllocaPair : BaseInstAllocaMap) {
    auto *IP = UniformLoadTerm;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Load base value from Alloca " << *InstAllocaPair.second
                       << " in " << IP->getParent()->getName() << "\n";
    auto *Load = utils::loadFromAlloca(InstAllocaPair.second, NewContIdx, IP,
                                       InstAllocaPair.first->getName());
    LoadToAlloca[Load] = InstAllocaPair.second;
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

llvm::SmallVector<llvm::Instruction *, 16> SubCFG::topoSortInstructions(
    const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const {
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
  auto *LoadBB = llvm::BasicBlock::Create(OuterMostHeader->getContext(),
                                          "uniloadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                                          OuterMostHeader->getParent(), OuterMostHeader);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(OuterMostHeader);
  return LoadBB;
}

llvm::BasicBlock *SubCFG::createLoadBB(llvm::ValueToValueMapTy &VMap) {
  assert(EntryBB_);
  assert(VMap[EntryBB_]);
  auto *NewEntry = llvm::cast<llvm::BasicBlock>(static_cast<llvm::Value *>(VMap[EntryBB_]));
  auto *LoadBB = llvm::BasicBlock::Create(NewEntry->getContext(),
                                          "loadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                                          NewEntry->getParent(), NewEntry);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(NewEntry);
  return LoadBB;
}

// if the kernel contained a loop, it is possible, that values inside a single
// subcfg don't dominate their uses inside the same subcfg. This function
// identifies and fixes those values.
void SubCFG::fixSingleSubCfgValues(
    llvm::DominatorTree &DT,
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
    llvm::Value *ReqdArrayElements, VectorizationInfo &VecInfo,
    llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &LoadToAlloca) {

  auto *AllocaIP = LoadBB_->getParent()->getEntryBlock().getTerminator();
  auto *LoadIP = LoadBB_->getTerminator();
  auto *UniLoadIP = PreHeader_->getTerminator();
  llvm::IRBuilder Builder{LoadIP};

  llvm::DenseMap<llvm::Instruction *, llvm::Instruction *> InstLoadMap;
  llvm::Value *ContiguousIdx = HI.ContiguousIdx;

  for (auto *BB : NewBlocks_) {
    llvm::SmallVector<llvm::Instruction *, 16> Insts{};
    std::transform(BB->begin(), BB->end(), std::back_inserter(Insts), [](auto &I) { return &I; });
    for (auto *Inst : Insts) {
      auto &I = *Inst;
      for (auto *OPV : I.operand_values()) {
        // check if all operands dominate the instruction -> otherwise we have to fix it
        if (auto *OPI = llvm::dyn_cast<llvm::Instruction>(OPV); OPI && !DT.dominates(OPI, &I)) {
          if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(Inst)) {
            // if a PHI node, we have to check that the incoming values dominate the terminators
            // of the incoming block..
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
          HIPSYCL_DEBUG_ERROR << "Instruction not dominated " << I << " operand: " << *OPI << "\n";

          if (auto *Load = InstLoadMap.lookup(OPI))
            // if the already inserted Load does not dominate I, we must create another load.
            if (DT.dominates(Load, &I)) {
              I.replaceUsesOfWith(OPI, Load);
              continue;
            }

          if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(OPI))
            if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
              // TODO what should happen, if F is the sub group function and the gep indexes is a
              // work group "array". ContiguousIdx is the subgroup induction variable.
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
            if (auto *NewAlloca = utils::getLoopStateAllocaForLoad(*LInst))
              Alloca = NewAlloca;
          if (!Alloca) {
            HIPSYCL_DEBUG_INFO << "[SubCFG] No alloca, yet for " << *OPI << "\n";
            Alloca = utils::arrayifyInstruction(
                AllocaIP, OPI, ContiguousIdx,
                VecInfo.getVectorShape(I).isUniform() ? nullptr : ReqdArrayElements);
            VecInfo.setVectorShape(*Alloca, VecInfo.getVectorShape(I));
          }

          auto Idx = ContiguousIdx;
#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          // in split loop, OPI might be used multiple times, get the user, dominating this user
          // and insert load there
          llvm::Instruction *NewIP = &I;
          for (auto *U : OPI->users()) {
            if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U); UI && DT.dominates(UI, NewIP)) {
              NewIP = UI;
            }
          }
#else
          // doesn't happen if we keep the PHIs
          auto *NewIP = LoadIP;
          if (!Alloca->isArrayAllocation()) {
            NewIP = UniLoadIP;
            Idx = llvm::ConstantInt::get(ContiguousIdx->getType(), 0);
          }
#endif

          auto *Load = utils::loadFromAlloca(Alloca, Idx, NewIP, OPI->getName());
          LoadToAlloca[Load] = Alloca;

          utils::copyDgbValues(OPI, Load, NewIP);

#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          I.replaceUsesOfWith(OPI, Load);
          InstLoadMap.insert({OPI, Load});
#else
          // if a loop is conditionally split, the first block in a subcfg might have another
          // incoming edge, need to insert a PHI node then
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
} // namespace hipsycl::compiler::cbs

namespace { 
llvm::BasicBlock *createUnreachableBlock(llvm::Function &F) {
  auto *Default = llvm::BasicBlock::Create(F.getContext(), "cbs.while.default", &F);
  llvm::IRBuilder Builder{Default, Default->getFirstInsertionPt()};
  Builder.CreateUnreachable();
  return Default;
}

// create the actual while loop around the subcfgs and the switch instruction to
// select the next subCFG based on the value in \a LastBarrierIdStorage
llvm::BasicBlock *generateWhileSwitchAround(llvm::BasicBlock *PreHeader, llvm::BasicBlock *OldEntry,
                                            llvm::BasicBlock *Exit,
                                            llvm::AllocaInst *LastBarrierIdStorage,
                                            std::vector<SubCFG> &SubCFGs) {
  auto &F = *PreHeader->getParent();
  auto &M = *F.getParent();
  const auto &DL = M.getDataLayout();

  auto *WhileHeader = llvm::BasicBlock::Create(PreHeader->getContext(), "cbs.while.header",
                                               PreHeader->getParent(), OldEntry);
  llvm::IRBuilder Builder{WhileHeader, WhileHeader->getFirstInsertionPt()};
  auto *LastID = Builder.CreateLoad(LastBarrierIdStorage->getAllocatedType(), LastBarrierIdStorage,
                                    "cbs.while.last_barr.load");
  auto *Switch = Builder.CreateSwitch(LastID, createUnreachableBlock(F), SubCFGs.size());
  for (auto &Cfg : SubCFGs) {
    Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), Cfg.getEntryId()),
                    Cfg.getEntry());
    Cfg.getEntry()->replacePhiUsesWith(PreHeader, WhileHeader);
    Cfg.getExit()->getTerminator()->replaceSuccessorWith(Exit, WhileHeader);
  }
  Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), ExitBarrierId), Exit);

  Builder.SetInsertPoint(PreHeader->getTerminator());
  auto *MDAlloca = llvm::MDNode::get(
      F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::LoopState)});
  auto* Store = Builder.CreateStore(
      llvm::ConstantInt::get(LastBarrierIdStorage->getAllocatedType(), EntryBarrierId),
      LastBarrierIdStorage);
  Store->setMetadata(MDKind::Arrayified, MDAlloca);

  PreHeader->getTerminator()->replaceSuccessorWith(OldEntry, WhileHeader);
  return WhileHeader;
}

// drops all lifetime intrinsics - they are misinforming ASAN otherwise (and are
// not really fixable at the right scope..)
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
}

// fills \a Hull with all transitive users of \a Alloca
void fillUserHull(llvm::Value *Alloca, llvm::SmallVectorImpl<llvm::Instruction *> &Hull) {
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

// checks if all uses of an alloca are in just a single subcfg (doesn't have to be arrayified!)
std::optional<SubCFG*> isAllocaSubCfgInternal(llvm::Value *Alloca, std::vector<SubCFG> &SubCfgs,
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
    llvm::SmallPtrSet<llvm::BasicBlock *, 8> SubCfgSet{SubCfg.getNewBlocks().begin(),
                                                       SubCfg.getNewBlocks().end()};
    if (std::any_of(UserBlocks.begin(), UserBlocks.end(),
                    [&SubCfgSet](auto *BB) { return SubCfgSet.contains(BB); })) {
      if (!std::all_of(UserBlocks.begin(), UserBlocks.end(), [&SubCfgSet, Alloca](auto *BB) {
          if (SubCfgSet.contains(BB)) {
            return true;
          }
          HIPSYCL_DEBUG_INFO << "[SubCFG] BB not in subcfgset: " << BB->getName()
                             << " for alloca: ";
          HIPSYCL_DEBUG_EXECUTE_INFO(Alloca->print(llvm::outs()); llvm::outs() << "\n";)
          return false;
        })) {
           return std::nullopt;
      } else {
        return &SubCfg;
      }
    }
  }

  return nullptr;
}

llvm::GetElementPtrInst * createGEP(llvm::AllocaInst *Alloca,
                                       llvm::Instruction *Before,
                                       bool AlignPadding, llvm::Value* ContIndex) {

  std::vector<llvm::Value *> GEPArgs;
  GEPArgs.push_back(ContIndex);

  if (AlignPadding)
    GEPArgs.push_back(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(Alloca->getParent()->getContext()), 0));

  llvm::IRBuilder<> Builder(Before);

  llvm::GetElementPtrInst *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Builder.CreateGEP(
      Alloca->getAllocatedType(), Alloca, GEPArgs));

  assert(GEP != nullptr);

  return GEP;
}

// Widens the allocas in the entry block to array allocas.
// Replace uses of the original alloca with GEP that indexes the new alloca with
// \a Idx.
void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT,
                     std::vector<SubCFG> &SubCfgs, llvm::Value *ReqdArrayElements,
                     VectorizationInfo &VecInfo, llvm::Function &F, HierarchicalSplitInfo HI) {
  auto *MDAlloca = llvm::MDNode::get(
      EntryBlock->getContext(), {llvm::MDString::get(EntryBlock->getContext(), MDKind::LoopState)});

  llvm::SmallPtrSet<llvm::BasicBlock *, 32> SubCfgsBlocks;
  for (auto &SubCfg : SubCfgs)
    SubCfgsBlocks.insert(SubCfg.getNewBlocks().begin(), SubCfg.getNewBlocks().end());
  {
    llvm::SmallVector<llvm::AllocaInst *, 8> WL;
    llvm::SmallVector<llvm::AllocaInst *, 8> WLSubCfgInternal;
    for (auto &I : *EntryBlock) {
      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
        if (Alloca->hasMetadata(MDKind::Arrayified))
          continue; // already arrayified
        if (utils::anyOfUsers<llvm::Instruction>(Alloca, [&SubCfgsBlocks](llvm::Instruction *UI) {
              return !SubCfgsBlocks.contains(UI->getParent());
            }))
          continue;

        if (auto shape = VecInfo.getVectorShape(*Alloca); shape.isUniform()) {
          continue;
        }
        if (auto SubCfg = isAllocaSubCfgInternal(Alloca, SubCfgs, DT)) {
          if (*SubCfg) {
#if USE_RV
            WLSubCfgInternal.push_back(Alloca);
#else
            if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP || HI.Level == HierarchicalLevel::CBS) {
              auto *MDWorkItemLoop = llvm::MDNode::get(
                  F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::WorkItemLoop)});
              auto *MDAllocaProblem = llvm::MDNode::get(
                  F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::AllocaProblem)});
              auto *LoopId = llvm::makePostTransformationMetadata(F.getContext(), nullptr, {},
                                                                  {MDWorkItemLoop, MDAllocaProblem});
              (*SubCfg)->WILoopLatch->getTerminator()->setMetadata("llvm.loop", LoopId);
            }
#endif
          }
        } else {
          WL.push_back(Alloca);
        }
      }
    }

    for (auto *I : WLSubCfgInternal) {
      for (auto &SubCfg : SubCfgs) {
        llvm::IRBuilder AllocaBuilder{SubCfg.getLoadBB()->getFirstNonPHI()};
        auto* AllocaClone = I->clone();
        AllocaBuilder.Insert(AllocaClone);
        llvm::replaceDominatedUsesWith(I, AllocaClone, DT, SubCfg.getLoadBB());
      }
      I->eraseFromParent();
    }

    for (auto *I : WL) {
      llvm::IRBuilder AllocaBuilder{I};
      llvm::Type *AllocType = I->getAllocatedType();
      if (auto *ArrSizeC = llvm::dyn_cast<llvm::ConstantInt>(I->getArraySize())) {
        auto ArrSize = ArrSizeC->getLimitedValue();
        if (ArrSize > 1) {
          AllocType = llvm::ArrayType::get(AllocType, ArrSize);
          llvm::outs() << "Caution, alloca was array\n";
        }
      }

      // Moved in from POCL

      auto *ElementType = I->getAllocatedType();
      auto &layout = F.getParent()->getDataLayout();

      unsigned Alignment = I->getAlign().value();
      uint64_t StoreSize = layout.getTypeStoreSize(ElementType);
      bool PaddingAdded = false;

      if ((Alignment > 1) && (StoreSize & (Alignment - 1))) {
        uint64_t AlignedSize = (StoreSize & (~(Alignment - 1))) + Alignment;

        assert(AlignedSize > StoreSize);
        uint64_t RequiredExtraBytes = AlignedSize - StoreSize;

        // n-dim context array: In case the elementType itself is an array or
        // a struct, we must take into account it could be alloca-ed with
        // alignment and loads or stores might use vectorized instructions
        // expecting proper alignment.
        // Because of that, we cannot simply allocate vectorWidth*(size), but must
        // pad the inner row to ensure the alignment to the next element.
        if (llvm::isa<llvm::ArrayType>(ElementType)) {

          llvm::ArrayType *StructPadding =
              llvm::ArrayType::get(llvm::Type::getInt8Ty(F.getContext()), RequiredExtraBytes);

          std::vector<llvm::Type *> PaddedStructElements;
          PaddedStructElements.push_back(ElementType);
          PaddedStructElements.push_back(StructPadding);
          const llvm::ArrayRef<llvm::Type *> NewStructElements(PaddedStructElements);
          AllocType = llvm::StructType::get(F.getContext(), NewStructElements, true);
          PaddingAdded = true;
          uint64_t NewStoreSize = layout.getTypeStoreSize(AllocType);
          assert(NewStoreSize == AlignedSize);

        } else if (llvm::isa<llvm::StructType>(ElementType)) {
          llvm::StructType *OldStruct = llvm::dyn_cast<llvm::StructType>(ElementType);

          llvm::ArrayType *StructPadding =
              llvm::ArrayType::get(llvm::Type::getInt8Ty(F.getContext()), RequiredExtraBytes);
          std::vector<llvm::Type *> PaddedStructElements;
          for (unsigned j = 0; j < OldStruct->getNumElements(); j++)
            PaddedStructElements.push_back(OldStruct->getElementType(j));
          PaddedStructElements.push_back(StructPadding);
          const llvm::ArrayRef<llvm::Type *> NewStructElements(PaddedStructElements);
          AllocType = llvm::StructType::get(OldStruct->getContext(), NewStructElements,
                                            OldStruct->isPacked());
          PaddingAdded = true;
          uint64_t NewStoreSize = layout.getTypeStoreSize(AllocType);
          assert(NewStoreSize == AlignedSize);
        }
      }

      auto *Alloca = AllocaBuilder.CreateAlloca(AllocType, ReqdArrayElements, I->getName() + "_alloca");
      Alloca->setAlignment(llvm::Align{DefaultAlignment});
      Alloca->setMetadata(MDKind::Arrayified, MDAlloca);

      for (auto &SubCfg : SubCfgs) {
        auto *GepIp = SubCfg.getLoadBB()->getFirstNonPHIOrDbgOrLifetime();
        auto *ContiguousIdx = SubCfg.getHI().ContiguousIdx;

        llvm::IRBuilder LoadBuilder{GepIp};
        auto* GEP = createGEP(Alloca, GepIp, PaddingAdded, ContiguousIdx);
        GEP->setMetadata(MDKind::Arrayified, MDAlloca);

        llvm::replaceDominatedUsesWith(I, GEP, DT, SubCfg.getLoadBB());
      }
      I->eraseFromParent();
    }
  }


  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    auto NumUses = mergeGVLoadsInEntry(F, cbs::SgLocalIdGlobalName)->getNumUses();
    assert(NumUses == 0);
  }
}

void moveAllocasToEntry(llvm::Function &F, llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  llvm::SmallVector<llvm::AllocaInst *, 4> AllocaWL;
  for (auto *BB : Blocks)
    for (auto &I : *BB)
      if (auto *AllocaInst = llvm::dyn_cast<llvm::AllocaInst>(&I))
        AllocaWL.push_back(AllocaInst);
  for (auto *I : AllocaWL)
    I->moveBefore(F.getEntryBlock().getTerminator());
}

llvm::DenseMap<llvm::BasicBlock *, size_t>
getBarrierIds(llvm::BasicBlock *Entry, llvm::SmallPtrSetImpl<llvm::BasicBlock *> &ExitingBlocks,
              llvm::ArrayRef<llvm::BasicBlock *> Blocks, const SplitterAnnotationInfo &SAA,
              HierarchicalLevel HL) {
  llvm::DenseMap<llvm::BasicBlock *, size_t> Barriers;
  // mark exit barrier with the corresponding id:
  for (auto *BB : ExitingBlocks)
    Barriers[BB] = ExitBarrierId;
  // mark entry barrier with the corresponding id:
  Barriers[Entry] = EntryBarrierId;

  // store all other barrier blocks with a unique id:
  size_t BarrierId = 1;
  auto hasOnlyBarrier =
      HL == HierarchicalLevel::H_CBS_SUBGROUP ? utils::hasOnlySubBarrier : utils::hasOnlyBarrier;

  for (auto *BB : Blocks)
    if (Barriers.find(BB) == Barriers.end() && hasOnlyBarrier(BB, SAA))
      Barriers.insert({BB, BarrierId++});
  return Barriers;
}

void formSubCfgGeneric(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                       llvm::PostDominatorTree &PDT, const SplitterAnnotationInfo &SAA, State state,
                       llvm::ArrayRef<llvm::Value *> LocalSize, llvm::Value *ReqdArrayElements,
                       HierarchicalSplitInfo HI);

void formSubgroupCfgs(SubCFG &Cfg, llvm::Function &F, const SplitterAnnotationInfo &SAA,
                      llvm::ArrayRef<llvm::Value *> LocalSize, State state,
                      HierarchicalSplitInfo HI, std::vector<SubCFG> &SubCfgs, llvm::DominatorTree& DT) {
  auto Blocks = Cfg.getNewBlocks(); // copy
  Blocks.insert(Blocks.begin(), Cfg.getLoadBB());

  auto innerIdx = llvm::dyn_cast<llvm::Instruction>(Cfg.getHI().ContiguousIdx)->getOperand(0);

  assert(!llvm::verifyFunction(F, &llvm::outs()));
  llvm::SetVector<llvm::Value *> Inputs, Outputs;
  llvm::CodeExtractorAnalysisCache CEAC{F};
  llvm::CodeExtractor CE{Blocks};
  assert(CE.isEligible());
  auto *NewF = CE.extractCodeRegion(CEAC, Inputs, Outputs);

  // Create a mapping between the old instructions (which now may have become
  // function arguments)
  // InAndOutToArgs: Old Instruction -> Function Argument
  llvm::ValueToValueMapTy InAndOutToArgs;
  llvm::SmallDenseMap<llvm::Argument *, llvm::AllocaInst *, 8> ArgsToAlloca;
  {
    HIPSYCL_DEBUG_INFO << "Inputs:"
                       << "\n";
    int Cnter = 0;
    for (auto I : Inputs) {
      HIPSYCL_DEBUG_INFO << *I << " -> " << *NewF->getArg(Cnter) << "\n";
      InAndOutToArgs[I] = NewF->getArg(Cnter);
      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(I)) {
        ArgsToAlloca[NewF->getArg(Cnter)] = Alloca;
      }
      Cnter++;
    }
    HIPSYCL_DEBUG_INFO << "Outputs:"
                       << "\n";
    for (auto O : Outputs) {
      HIPSYCL_DEBUG_INFO << *O << " -> " << *NewF->getArg(Cnter) << "\n";
      InAndOutToArgs[O] = NewF->getArg(Cnter);
      if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(O)) {
        ArgsToAlloca[NewF->getArg(Cnter)] = Alloca;
      }
      Cnter++;
    }
  }

  // If alloca is only used in SUBGROUPS that are between the same work group barriers, then
  // the alloca is not arrayified on the work group level
  {
    llvm::IRBuilder Builder{NewF->getEntryBlock().getFirstNonPHI()};
    for (auto [Arg, Alloca] : ArgsToAlloca) {
      if (Alloca->hasMetadata(MDKind::Arrayified))
        continue;
      if (Alloca->getNumUses() > 1)
        continue;
      auto AllocaClone = Alloca->clone();
      auto Inserted = Builder.Insert(AllocaClone);
      assert(Inserted == AllocaClone);
      Arg->replaceAllUsesWith(AllocaClone);
      Alloca->replaceAllUsesWith(llvm::UndefValue::get(Alloca->getType()));
      Alloca->eraseFromParent();
      assert(Arg->getNumUses() == 0);
    }
  }
  // Create barriers at the beginning and end of the cfg
  {
    utils::createSubBarrier(NewF->getEntryBlock().getTerminator(),
                            const_cast<SplitterAnnotationInfo &>(SAA));
    for (auto &BB : *NewF)
      if (BB.getTerminator()->getNumSuccessors() == 0)
        utils::createSubBarrier(BB.getTerminator(), const_cast<SplitterAnnotationInfo &>(SAA));
  }

  llvm::DominatorTree NewDT{*NewF};
  llvm::PostDominatorTree NewPDT{*NewF};
  llvm::LoopInfo NewLI{NewDT};


  llvm::Value *InnerSize = [&]() -> llvm::Value* {
    if (auto it = InAndOutToArgs.find(LocalSize.back()); it != InAndOutToArgs.end()) {
      return it->second;
    }
    return mergeGVLoadsInEntry(*NewF, state.LocalSizeGlobalNames[LocalSize.size() - 1],
                               LocalSize.back()->getType());
  }();

  auto replaceInNewF = [&](const std::string& globalVarName) -> llvm::Value * {
    auto* V = mergeGVLoadsInEntry(F, globalVarName);
    llvm::Value* VinNewF = mergeGVLoadsInEntry(*NewF, globalVarName, V->getType());
    if (auto It = InAndOutToArgs.find(V); It != InAndOutToArgs.end()) {
      HIPSYCL_DEBUG_INFO << "Replace " << *V << " with " << *VinNewF << "\n";
      It->second->replaceAllUsesWith(VinNewF);
    }
    return VinNewF;
  };

  llvm::Value *SGIdArg = replaceInNewF(cbs::SgLocalIdGlobalName);
  llvm::Value *SGGroupIdArg = replaceInNewF(cbs::SgIdGlobalName);

  formSubCfgGeneric(*NewF, NewLI, NewDT, NewPDT, SAA, state,
                    {llvm::ConstantInt::get(LocalSize[0]->getType(), SGSize)},
                    llvm::ConstantInt::get(LocalSize[0]->getType(), SGSize),
                    {HierarchicalLevel::H_CBS_SUBGROUP, SGIdArg, &ArgsToAlloca, InnerSize});

  // The SgIdArg in NewF should not have any users.
  // They should have been replaced with the subgroup induction variable
  assert(SGIdArg->getNumUses() == 0);
  assert(SGGroupIdArg->getNumUses() == 0);

  assert(std::distance(NewF->user_begin(), NewF->user_end()) == 1);
  utils::checkedInlineFunction(llvm::cast<llvm::CallBase>(NewF->user_back()), PASS_PREFIX_STR);
  NewF->eraseFromParent();

  utils::replaceUsesOfGVWith(F, "__cont_idx_without_sg", innerIdx, PASS_PREFIX_STR);
  utils::replaceUsesOfGVWith(F, "__inner_ind_var", Cfg.getInnerPhiIndVar(), PASS_PREFIX_STR);
}

void formSubCfgGeneric(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                       llvm::PostDominatorTree &PDT, const SplitterAnnotationInfo &SAA, State state,
                       llvm::ArrayRef<llvm::Value *> LocalSize, llvm::Value *ReqdArrayElements,
                       HierarchicalSplitInfo HI) {
  auto *Entry = &F.getEntryBlock();

  std::vector<llvm::BasicBlock *> Blocks;
  {
    Blocks.reserve(std::distance(F.begin(), F.end()));
    std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });
  }

  // non-entry block Allocas are considered broken, move to entry.
  moveAllocasToEntry(F, Blocks);
  mergeGVLoadsInEntry(F, cbs::SgLocalIdGlobalName);
  mergeGVLoadsInEntry(F, cbs::SgIdGlobalName);
  mergeGVLoadsInEntry(F, cbs::SgSizeGlobalName);

  for (size_t D = 0; D < 3; ++D) {
    mergeGVLoadsInEntry(F, cbs::LocalIdGlobalNames[D]);
    mergeGVLoadsInEntry(F, cbs::LocalSizeGlobalNames[D]);
    mergeGVLoadsInEntry(F, cbs::NumGroupsGlobalNames[D]);
    mergeGVLoadsInEntry(F, cbs::GroupIdGlobalNames[D]);
  }


  auto RImpl = getRegion(F, LI, Blocks);
  Region R{*RImpl};
  auto VecInfo = getVectorizationInfo(F, R, LI, DT, PDT, state.Dim, state, HI);

  llvm::SmallPtrSet<llvm::BasicBlock *, 2> ExitingBlocks;
  R.getEndingBlocks(ExitingBlocks);

  if (ExitingBlocks.empty()) {
    HIPSYCL_DEBUG_ERROR << "[SubCFG] Invalid kernel! No kernel exits!\n";
    llvm_unreachable("[SubCFG] Invalid kernel! No kernel exits!\n");
  }

  auto Barriers = getBarrierIds(Entry, ExitingBlocks, Blocks, SAA, HI.Level);

  llvm::IRBuilder Builder{F.getEntryBlock().getFirstNonPHI()};
  const llvm::DataLayout &DL = F.getParent()->getDataLayout();
  auto *MDAlloca = llvm::MDNode::get(
      F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::LoopState)});

  auto *LastBarrierIdStorage =
      Builder.CreateAlloca(DL.getLargestLegalIntType(F.getContext()), nullptr, "LastBarrierId");
  LastBarrierIdStorage->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

  // create subcfgs
  std::vector<SubCFG> SubCFGs;
  for (auto &[Barrier, Id] : Barriers) {
    HIPSYCL_DEBUG_INFO << "Create SubCFG from " << Barrier->getName() << "(" << Barrier
                       << ") id: " << Id << "\n";
    if (Id != ExitBarrierId) {
      if (!Barrier->getSingleSuccessor()) {
        if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
          llvm::outs() << "SUB\n";
        }
        llvm::outs() << F;
        llvm::outs() << "NSS: " << Barrier->getName() << "\n";
      }
      SubCFGs.emplace_back(Barrier, LastBarrierIdStorage, Barriers, SAA, state.Dim, HI);
    }
  }

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> InstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> BaseInstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> InstContReplicaMap;
  llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> LoadToAlloca{};

  for (auto &Cfg : SubCFGs)
    Cfg.arrayifyMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, SubCFGs,
                                  F.getEntryBlock().getTerminator(), ReqdArrayElements, VecInfo, F);

  llvm::BasicBlock *NewExit =
      llvm::BasicBlock::Create(F.getContext(), "cbs.exit", &F);
  llvm::IRBuilder<> ExitBuilder(NewExit);
  ExitBuilder.CreateRetVoid();

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> RemappedInstAllocaMap;
  // TODO remapped Instr alloca is used for fixSingleSubCfgValue
  for (auto &Cfg : SubCFGs) {
    Cfg.print();
    Cfg.replicate(F, InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, RemappedInstAllocaMap,
                  NewExit, LocalSize, state, LoadToAlloca);
    purgeLifetime(Cfg);
  }

  llvm::BasicBlock *WhileHeader =
      generateWhileSwitchAround(&F.getEntryBlock(), F.getEntryBlock().getSingleSuccessor(),
                                NewExit, LastBarrierIdStorage, SubCFGs);

  llvm::removeUnreachableBlocks(F);

  DT.recalculate(F);
  arrayifyAllocas(&F.getEntryBlock(), DT, SubCFGs, ReqdArrayElements, VecInfo, F, HI);

  for (auto &Cfg : SubCFGs) {
    Cfg.fixSingleSubCfgValues(DT, RemappedInstAllocaMap, ReqdArrayElements, VecInfo, LoadToAlloca);
  }

  //assert(mergeGVLoadsInEntry(F, cbs::SgIdGlobalName)->getNumUses() == 0);

  if (HI.Level == HierarchicalLevel::H_CBS_GROUP) {
    for (auto &Cfg : SubCFGs) {
      formSubgroupCfgs(Cfg, F, SAA, LocalSize, state, HI, SubCFGs, DT);
    }
  }

  if (HI.Level == HierarchicalLevel::H_CBS_SUBGROUP) {
    assert(!llvm::verifyFunction(F, &llvm::errs()));
    ReduceIntrinsic{}.vectorizeAllInstances(F, SubCFGs, &InstContReplicaMap);
    Shift<true>{}.vectorizeAllInstances(F, SubCFGs);
    Shift<false>{}.vectorizeAllInstances(F, SubCFGs);
    ShuffleIntrinsic{}.vectorizeAllInstances(F, SubCFGs);
    ExtractIntrinsic{}.vectorizeAllInstances(F, SubCFGs);
    assert(!llvm::verifyFunction(F, &llvm::outs()));
  }

  // simplify while loop to get single latch that isn't marked as wi-loop to prevent
  // misunderstandings.
  auto *WhileLoop = utils::updateDtAndLi(LI, DT, WhileHeader, F);
  llvm::simplifyLoop(WhileLoop, &DT, &LI, nullptr, nullptr, nullptr, false);
}

void formSubCfgs(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                 llvm::PostDominatorTree &PDT, const SplitterAnnotationInfo &SAA, State state) {
  HIPSYCL_DEBUG_ERROR << "[SubCFG] Kernel is " << state.Dim << "-dimensional\n";

  const auto LocalSize = getLocalSizeValues(F, state);
  auto *Entry = &F.getEntryBlock();
  llvm::IRBuilder Builder{Entry->getTerminator()};
  llvm::Value *ReqdArrayElements = LocalSize[0];
  for (size_t D = 1; D < LocalSize.size(); ++D)
    ReqdArrayElements = Builder.CreateMul(ReqdArrayElements, LocalSize[D]);

  // get a common (pseudo) index value to be replaced by the actual index later
  Builder.SetInsertPoint(F.getEntryBlock().getTerminator());
  llvm::Instruction *IndVar = Builder.CreateLoad(
      state.SizeT, llvm::UndefValue::get(llvm::PointerType::get(state.SizeT, 0)));

  if constexpr (USE_RV) {
    assert(!utils::hasSubBarriers(F, SAA));
  }

  const bool PerformHCBS = utils::hasSubBarriers(F, SAA) or ALWAYS_CREATE_SUBGROUP_SUB_CFGS;

  auto* SgNumSubgroups = mergeGVLoadsInEntry(F, cbs::SgNumSubgroupsGlobalName);

  formSubCfgGeneric(
      F, LI, DT, PDT, SAA, state, LocalSize, ReqdArrayElements,
      {PerformHCBS ? HierarchicalLevel::H_CBS_GROUP : HierarchicalLevel::CBS,
       IndVar});

  // Calculate SgNumSubgroups and replace cbs::SgNumSubgroupsGlobalName
  {
    auto *InnerMostSize = mergeGVLoadsInEntry(F, state.LocalSizeGlobalNames[state.Dim - 1]);
    // We round up
    auto *NumSgsInnerMostDimension = Builder.CreateUDiv(
        Builder.CreateAdd(InnerMostSize,
                          Builder.getIntN(state.SizeT->getIntegerBitWidth(), SGSize - 1)),
        Builder.getIntN(state.SizeT->getIntegerBitWidth(), SGSize));

    auto *NumSgs = NumSgsInnerMostDimension;
    for (auto i = 0ul; i < state.Dim - 1; ++i) {
      NumSgs = Builder.CreateMul(NumSgs, mergeGVLoadsInEntry(F, state.LocalSizeGlobalNames[i]));
    }
    SgNumSubgroups->replaceAllUsesWith(NumSgs);
  }

  assert(!llvm::verifyFunction(F, &llvm::errs()) && "Function verification failed");
}

void multiplyFunction(llvm::Function &F, State state) {
  auto& M = *F.getParent();
  auto& Context = M.getContext();
  std::string NewFunctionName = std::string{F.getName()}+"_multiply";
  auto* NewF = llvm::Function::Create(F.getFunctionType(), F.getLinkage(), F.getAddressSpace(), NewFunctionName, &M);
  NewF->setAttributes( F.getAttributes());

  assert(NewF->getParent());

  auto* entryBB = llvm::BasicBlock::Create(Context, "entry", NewF);
  auto* noIncompleteSgBB = llvm::BasicBlock::Create(Context, "then", NewF);
  auto* elseBB = llvm::BasicBlock::Create(Context, "else", NewF);
  auto* endBB = llvm::BasicBlock::Create(Context, "end", NewF);

  llvm::IRBuilder<> Builder{entryBB};
  auto* earlyRet = Builder.CreateRetVoid();


  Builder.CreateCondBr( mergeGVLoadsInEntry(*NewF, "no-incomplete-sgs", Builder.getInt1Ty()), noIncompleteSgBB, elseBB);

  Builder.SetInsertPoint(elseBB);
  Builder.CreateBr(endBB);

  Builder.SetInsertPoint(noIncompleteSgBB);
  Builder.CreateBr(endBB);

  Builder.SetInsertPoint(endBB);
  Builder.CreateRetVoid();

  earlyRet->eraseFromParent();

  llvm::SmallVector<llvm::Value*, 8> Args{};
  for (auto& Arg: NewF->args()) {
    Args.emplace_back(&Arg);
  }

  Builder.SetInsertPoint(noIncompleteSgBB->getFirstNonPHI());
  auto* CallComplete = Builder.CreateCall(F.getFunctionType(), &F, Args);
  assert(!llvm::verifyFunction(*NewF, &llvm::outs()));
  utils::checkedInlineFunction(CallComplete, "", true);

  Builder.SetInsertPoint(elseBB->getFirstNonPHI());
  auto* CallInComplete = Builder.CreateCall(F.getFunctionType(), &F, Args);
  assert(!llvm::verifyFunction(*NewF, &llvm::outs()));
  utils::checkedInlineFunction(CallInComplete, "", true);


  auto* newEntry = llvm::BasicBlock::Create(F.getContext(), "Newentry", &F);
  llvm::IRBuilder<> B{newEntry};
  B.CreateRetVoid();

  auto* origTerm = F.getEntryBlock().getTerminator();
  B.SetInsertPoint(origTerm);
  B.CreateBr(newEntry);
  origTerm->eraseFromParent();

  assert(!llvm::verifyFunction(F, &llvm::outs()));
  {
    llvm::SmallVector<llvm::Value*, 8> Args{};
    for (auto& Arg: F.args()) {
      Args.emplace_back(&Arg);
    }
    B.SetInsertPoint(newEntry->getFirstNonPHI());
    auto* CallNewF = B.CreateCall(NewF->getFunctionType(), NewF, Args);
    assert(!llvm::verifyFunction(F, &llvm::outs()));
    utils::checkedInlineFunction(CallNewF, "", true);
  }

  NewF->eraseFromParent();

  {
    llvm::IRBuilder<> Builder{F.getEntryBlock().getFirstNonPHI()};
    llvm::Value* InnerMostDimensionSize = mergeGVLoadsInEntry(F, state.LocalSizeGlobalNames[state.Dim-1]);
    llvm::Value* Cond =  Builder.CreateURem(InnerMostDimensionSize, Builder.getInt64(SGSize));
    auto* CondNoIncompleteSgs = Builder.CreateICmpEQ(Cond, Builder.getInt64(0));
    utils::replaceUsesOfGVWith(F, "no-incomplete-sgs", CondNoIncompleteSgs);
  }
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

  if (!SAA.isKernelFunc(&F) || getRangeDim(F) == 0)
    return false;

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &PDT = getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();

  assert(false);
  // if (utils::hasBarriers(F, SAA) || utils::hasSubBarriers(F, SAA))
  formSubCfgs(F, LI, DT, PDT, SAA, {});
  // else
  // createLoopsAroundKernel(F, DT, LI, PDT, {});

  return true;
}

char SubCfgFormationPassLegacy::ID = 0;

llvm::PreservedAnalyses SubCfgFormationPass::run(llvm::Function &F,
                                                 llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());

  if (!SAA || !SAA->isKernelFunc(&F) || getRangeDim(F) == 0)
    return llvm::PreservedAnalyses::all();

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  assert(!llvm::verifyFunction(F, &llvm::outs()));

  State state{.Dim = getRangeDim(F),
              .SizeT = llvm::IntegerType::getInt64Ty(F.getContext())};

  if (IsSscp_) {
    std::reverse(state.DimName.begin(), state.DimName.begin() + state.Dim);
    std::reverse(state.LocalSizeGlobalNames.begin(), state.LocalSizeGlobalNames.begin() + state.Dim);
    std::reverse(state.LocalIdGlobalNames.begin(), state.LocalIdGlobalNames.begin() + state.Dim);
  }

  const bool hasSubbarriers = utils::hasSubBarriers(F, *SAA);

  formSubCfgs(F, LI, DT, PDT, *SAA, state);

  if (!USE_RV && INCOMPLETE_SGS_OPT && (hasSubbarriers || ALWAYS_CREATE_SUBGROUP_SUB_CFGS)) {
    multiplyFunction(F, state);
  }

  if (!IsSscp_) {
    // If we do not use SSCP, then we need to replace localSizes
    // In SSCP, the LocalSizeGlobalNames are replaced in a later pipeline stage
    const auto LocalSizes = loadLocalSizesFromAnnotations(F, state);
    assert(LocalSizes.size() == state.Dim);
    for (auto i = 0ul; i < state.Dim; ++i) {
      utils::replaceUsesOfGVWith(F, state.LocalSizeGlobalNames[i], LocalSizes[i], PASS_PREFIX_STR);
    }
  }

  llvm::IRBuilder Builder{F.getContext()};

  for (auto i = state.Dim; i < 3; ++i) {
    utils::replaceUsesOfGVWith(F, state.LocalSizeGlobalNames[i], Builder.getIntN(64, 1), PASS_PREFIX_STR);
    utils::replaceUsesOfGVWith(F, state.LocalIdGlobalNames[i], Builder.getIntN(64, 0), PASS_PREFIX_STR);
  }

  for (auto i = 0; i < 3; ++i) {
    mergeGVLoadsInEntry(F, state.LocalSizeGlobalNames[i]);
  }


  // SSCP shared memory
  {
    Builder.SetInsertPoint(F.getEntryBlock().getFirstNonPHI());
    {
      auto* WorkgroupScratchMemoryAlloca = Builder.CreateAlloca(llvm::IntegerType::getInt8Ty(F.getContext()),
                                               Builder.getIntN(64, 1024 * 1024));
      WorkgroupScratchMemoryAlloca->setAlignment(llvm::Align(128));
      utils::replaceUsesOfGVWith(F, cbs::WorkGroupSharedMemory,
                          WorkgroupScratchMemoryAlloca, PASS_PREFIX_STR);
    }
    {
      auto* SubgroupScratchMemoryAlloca = Builder.CreateAlloca(llvm::IntegerType::getInt8Ty(F.getContext()),
                                               Builder.getIntN(64, 32 * 1024));
      SubgroupScratchMemoryAlloca->setAlignment(llvm::Align(128));
      utils::replaceUsesOfGVWith(F, cbs::SubGroupSharedMemory, SubgroupScratchMemoryAlloca, PASS_PREFIX_STR);
    }
  }
  F.addFnAttr(llvm::Attribute::NoInline);
  assert(!llvm::verifyFunction(F, &llvm::outs()));

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler