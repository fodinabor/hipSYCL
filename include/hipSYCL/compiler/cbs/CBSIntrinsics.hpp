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
#ifndef ACPP_CBSINTRINSICS_HPP
#define ACPP_CBSINTRINSICS_HPP

#include "hipSYCL/compiler/cbs/IRUtils.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <vector>
#include <numeric>

namespace llvm {
class Function;
}

namespace hipsycl {
namespace compiler {
namespace cbs {
class SubCFG;

class CBSIntrinsic {
public:
  virtual ~CBSIntrinsic() = default;
  void vectorizeAllInstances(
      llvm::Function &F, std::vector<SubCFG> &SubCfgs,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> *cont =
          nullptr);

protected:
  enum class Shape { VARYING, UNIFORM };

private:
  llvm::Value *vectorizeFunction(
      llvm::Function &F, llvm::CallInst &Intrinsic, SubCFG &SubCfg,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> *cont);

  static std::pair<llvm::Value *, Shape> getOrCreateValue(
      llvm::IRBuilder<> &Builder, const SubCFG &SubCfg, const llvm::CallInst &Intrinsic,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> *cont);

  static SubCFG &findSubCfg(std::vector<SubCFG> &SubCfgs, llvm::Instruction *I);

  static llvm::SmallVector<llvm::CallInst *, 8> getIntrinsic(llvm::Function &F,
                                                             std::string_view str);

  virtual std::string_view getName() = 0;
  virtual std::pair<llvm::Value *, Shape>
  vectorizeUniformValue(llvm::Value *Storage, llvm::IRBuilder<> &Builder, llvm::CallInst &Intrinsic,
                        llvm::Value *NumberOfLoopIterationsLeft) = 0;
  virtual std::pair<llvm::Value *, Shape> vectorizeValue(llvm::Instruction *VLoad,
                                                         llvm::IRBuilder<> &Builder,
                                                         llvm::CallInst &Intrinsic) = 0;
  virtual llvm::Value *extractElement(llvm::IRBuilder<> &Builder, llvm::Value *Value,
                                      llvm::Value *SgInductionVariable, llvm::CallInst &Intrinsic) {
    return Builder.CreateExtractElement(Value, SgInductionVariable);
  }

  virtual llvm::Value *neutralElement(llvm::VectorType *Type, llvm::IRBuilder<> &Builder,
                                      llvm::CallInst &Instrinsic) const {
    return llvm::ConstantInt::get(Type, 0);
  }
};

class ReduceIntrinsic final : public CBSIntrinsic {
  std::string_view getName() override { return "__cbs_reduce"; }

  std::string getTypeStr(llvm::Type *Type);

  std::pair<llvm::Value *, Shape>
  vectorizeUniformValue(llvm::Value *Storage, llvm::IRBuilder<> &Builder, llvm::CallInst &Intrinsic,
                        llvm::Value *NumberOfLoopIterationsLeft) override;

  std::pair<llvm::Value *, Shape> vectorizeValue(llvm::Instruction *VLoad,
                                                 llvm::IRBuilder<> &Builder,
                                                 llvm::CallInst &Intrinsic) override;

  llvm::Value *neutralElement(llvm::VectorType *Type, llvm::IRBuilder<> &Builder,
                              llvm::CallInst &Intrinsic) const override;

  llvm::Function *createPowFunction(llvm::Module *module, llvm::Type *Type);
};

template <bool Left> class Shift final : public CBSIntrinsic {
  std::string_view getName() override { return Left ? "__cbs_shift_left" : "__cbs_shift_right"; }
  std::pair<llvm::Value *, Shape>
  vectorizeUniformValue(llvm::Value *Storage, llvm::IRBuilder<> &Builder, llvm::CallInst &Intrinsic,
                        llvm::Value *NumberOfLoopIterationsLeft) override {
    return {Storage, Shape::UNIFORM};
  }

  std::pair<llvm::Value *, Shape> vectorizeValue(llvm::Instruction *VLoad,
                                                 llvm::IRBuilder<> &Builder,
                                                 llvm::CallInst &Intrinsic) override {
    auto *Idx = Intrinsic.getOperand(1);
    assert(Idx && "Op must be constant int");
    if (const auto *Op1V = llvm::dyn_cast<llvm::ConstantInt>(Idx)) {
      std::array<int, SGSize> mask{};
      {
        std::iota(mask.begin(), mask.end(), 0);
        for (auto &x : mask) {
          x = x + Op1V->getSExtValue();
          if (x >= SGSize) {
            x = 0;
          }
        }
      }
      return {Builder.CreateShuffleVector(VLoad, mask), Shape::VARYING};
    } else {
      return {VLoad, Shape::UNIFORM};
    }
  }

  llvm::Value *extractElement(llvm::IRBuilder<> &Builder, llvm::Value *Value,
                              llvm::Value *SgInductionVariable,
                              llvm::CallInst &Intrinsic) override {
    auto *Op1 = Intrinsic.getOperand(1);
    auto *Idx = Builder.CreateURem(
        Left ? Builder.CreateAdd(SgInductionVariable, Op1)
             : Builder.CreateSub(SgInductionVariable, Op1),
        llvm::ConstantInt::get(
            Builder.getContext(),
            llvm::APInt(SgInductionVariable->getType()->getIntegerBitWidth(), SGSize)));

    return Builder.CreateExtractElement(Value, Idx);
  }
};

class ShuffleIntrinsic final : public CBSIntrinsic {
  std::string_view getName() override { return "__cbs_shuffle"; }

  std::pair<llvm::Value *, Shape>
  vectorizeUniformValue(llvm::Value *Storage, llvm::IRBuilder<> &Builder, llvm::CallInst &Intrinsic,
                        llvm::Value *NumberOfLoopIterationsLeft) override {
    return {Storage, Shape::UNIFORM};
  }

  std::pair<llvm::Value *, Shape> vectorizeValue(llvm::Instruction *VLoad,
                                                 llvm::IRBuilder<> &Builder,
                                                 llvm::CallInst &Intrinsic) override;

  llvm::Value *extractElement(llvm::IRBuilder<> &Builder, llvm::Value *Value,
                              llvm::Value *SgInductionVariable, llvm::CallInst &Intrinsic) override;
};

class ExtractIntrinsic final : public CBSIntrinsic {
  std::string_view getName() override { return "__cbs_extract"; }

  std::pair<llvm::Value *, Shape>
  vectorizeUniformValue(llvm::Value *Storage, llvm::IRBuilder<> &Builder, llvm::CallInst &Intrinsic,
                        llvm::Value *NumberOfLoopIterationsLeft) override;

  std::pair<llvm::Value *, Shape> vectorizeValue(llvm::Instruction *VLoad,
                                                 llvm::IRBuilder<> &Builder,
                                                 llvm::CallInst &Intrinsic) override;
};
} // namespace cbs
} // namespace compiler
} // namespace hipsycl
#endif