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
#ifndef HIPSYCL_SUBCFGFORMATION_HPP
#define HIPSYCL_SUBCFGFORMATION_HPP

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace hipsycl {
namespace compiler {

constexpr size_t EntryBarrierId = 0;
constexpr size_t ExitBarrierId = -1;

// performs the main CBS transformation
class SubCfgFormationPassLegacy : public llvm::FunctionPass {
public:
  static char ID;

  explicit SubCfgFormationPassLegacy() : llvm::FunctionPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL sub-CFG formation pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;
};

class SubCfgFormationPass : public llvm::PassInfoMixin<SubCfgFormationPass> {
  bool IsSscp_;

public:
  explicit SubCfgFormationPass(bool IsSscp) : IsSscp_(IsSscp) {}

  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

struct SplitterAnnotationInfo;
class VectorizationInfo;

namespace cbs {
struct State;

enum class HierarchicalLevel {
  CBS,
  H_CBS_GROUP,
  H_CBS_SUBGROUP,
};

// Reference type only!
struct HierarchicalSplitInfo {
  HierarchicalLevel Level;
  llvm::Value *ContiguousIdx;
  // Are only set when we are in the sub-group level
  llvm::SmallDenseMap<llvm::Argument *, llvm::AllocaInst *, 8> *ArgsToAloca{};
  // Size of inner most work-group dimension
  llvm::Value *InnerSize{};
  // Innermost induction variable with sg-size stride
  llvm::Value *InnerPhiInd{};
};

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
  size_t Dim;
  HierarchicalSplitInfo HI;
  llvm::SmallVector<llvm::PHINode *, 3> WIPhiIndVars_;

  llvm::BasicBlock *
  createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
                   llvm::BasicBlock *After, llvm::BasicBlock *TargetBB);

  void loadMultiSubCfgValues(
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap,
      llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca);

  void loadUniformAndRecalcContValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap,
      llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca, State state);

  llvm::BasicBlock *createLoadBB(llvm::ValueToValueMapTy &VMap);

  llvm::BasicBlock *createUniformLoadBB(llvm::BasicBlock *OuterMostHeader);

public:
  llvm::BasicBlock *WILoopLatch;
  SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
         const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds,
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

  llvm::PHINode *getInnerPhiIndVar() const noexcept { return WIPhiIndVars_.back(); }
  HierarchicalSplitInfo getHI() const noexcept { return HI; }

  void replicate(llvm::Function &F,
                 const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
                     &ContInstReplicaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
                 llvm::BasicBlock *AfterBB, llvm::ArrayRef<llvm::Value *> LocalSize, State state,
                 llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca);

  void arrayifyMultiSubCfgValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>>
          &ContInstReplicaMap,
      llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, llvm::Value *ReqdArrayElements,
      VectorizationInfo &VecInfo, llvm::Function &F);

  void fixSingleSubCfgValues(
      llvm::DominatorTree &DT,
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
      llvm::Value *ReqdArrayElements, VectorizationInfo &VecInfo,
      llvm::DenseMap<llvm::LoadInst *, llvm::AllocaInst *> &loadToAlloca);

  void print() const;

  void removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const;

  llvm::SmallVector<llvm::Instruction *, 16>
  topoSortInstructions(const llvm::SmallPtrSet<llvm::Instruction *, 16> &UniquifyInsts) const;
};

llvm::LoadInst *mergeGVLoadsInEntry(llvm::Function &F, llvm::StringRef VarName,
  llvm::Type *ty = nullptr);

} // namespace cbs
} // namespace compiler
} // namespace hipsycl

#endif // HIPSYCL_SUBCFGFORMATION_HPP
