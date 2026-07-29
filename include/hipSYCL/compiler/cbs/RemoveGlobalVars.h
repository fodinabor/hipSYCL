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
#ifndef REMOVEGLOBALVARS_H
#define REMOVEGLOBALVARS_H

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace compiler {

// After SubCfgFormation has lowered all uses inside kernels, this pass cleans up the
// remaining references to the CBS pseudo global variables and intrinsics. Such references
// only survive in leftover, never-called copies of the (linkonce_odr) libkernel functions
// that were force-inlined into the kernels; without this cleanup they would produce
// undefined-symbol errors at host link time.
class RemoveGlobalVars
    : public llvm::PassInfoMixin<RemoveGlobalVars> {
public:
  explicit RemoveGlobalVars() = default;

  void removeGlobalVar(llvm::Module *M, llvm::StringRef VarName) {
    if (auto *GV = M->getGlobalVariable(VarName)) {
      llvm::SmallVector<llvm::Instruction *, 8> WL;
      for (auto U : GV->users())
        if (auto LI = llvm::dyn_cast<llvm::LoadInst>(U))
          WL.push_back(LI);

      for (auto *LI : WL) {
        LI->replaceAllUsesWith(llvm::PoisonValue::get(LI->getType()));
        LI->eraseFromParent();
      }

      if (GV->getNumUses() == 0 ||
          std::none_of(GV->user_begin(), GV->user_end(), [GV](llvm::User *U) { return U != GV; })) {
        HIPSYCL_DEBUG_INFO << "[RemoveGlobalVars] Clean-up global variable " << *GV << "\n";
        GV->eraseFromParent();
        return;
      }

      HIPSYCL_DEBUG_INFO << "[RemoveGlobalVars] Global variable still in use " << VarName << "\n";
      for (auto *U : GV->users()) {
        HIPSYCL_DEBUG_INFO << "[RemoveGlobalVars] >>> " << *U;
        if (auto I = llvm::dyn_cast<llvm::Instruction>(U)) {
          HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << " in " << I->getFunction()->getName(););
        }
      }
    }
  }

  // Leftover calls to the __cbs_* intrinsics (reduce/shuffle/broadcast/...) are replaced
  // with their data operand. This is only correct for sub-groups of size 1, but the only
  // surviving calls are in dead copies of functions that were inlined into the kernels.
  void removeCBSIntrinsics(llvm::Module *M) {
    llvm::SmallVector<llvm::Function *, 8> Intrinsics;
    for (auto &F : *M)
      if (F.isDeclaration() && utils::isCBSIntrinsic(&F))
        Intrinsics.push_back(&F);

    for (auto *F : Intrinsics) {
      llvm::SmallVector<llvm::CallInst *, 8> Calls;
      for (auto *U : F->users())
        if (auto *CI = llvm::dyn_cast<llvm::CallInst>(U); CI && CI->getCalledFunction() == F)
          Calls.push_back(CI);

      for (auto *CI : Calls) {
        HIPSYCL_DEBUG_INFO << "[RemoveGlobalVars] Replace leftover CBS intrinsic call " << *CI
                           << "\n";
        CI->replaceAllUsesWith(CI->getArgOperand(0));
        CI->eraseFromParent();
      }

      if (F->use_empty())
        F->eraseFromParent();
    }
  }

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
    removeGlobalVar(&M, cbs::SgLocalIdGlobalName);
    removeGlobalVar(&M, cbs::SgIdGlobalName);
    removeGlobalVar(&M, cbs::SgSizeGlobalName);
    removeGlobalVar(&M, cbs::SgNumSubgroupsGlobalName);
    removeGlobalVar(&M, cbs::SubGroupSharedMemory);
    removeGlobalVar(&M, cbs::WorkGroupSharedMemory);
    removeCBSIntrinsics(&M);
    return llvm::PreservedAnalyses::none();
  }

  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl

#endif //REMOVEGLOBALVARS_H
