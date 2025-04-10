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

#include <llvm/IR/LegacyPassManagers.h>
#include <llvm/IR/PassManager.h>
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace compiler {

/*
class RemoveGlobalVarsLegacy : public llvm::FunctionPass {
public:
  static char ID;
  explicit RemoveGlobalVarsLegacy() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) {

  }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override { AU.setPreservesAll(); }
};
*/

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

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
    removeGlobalVar(&M, cbs::SgLocalIdGlobalName);
    removeGlobalVar(&M, cbs::SgIdGlobalName);
    removeGlobalVar(&M, cbs::SubGroupSharedMemory);
    removeGlobalVar(&M, cbs::WorkGroupSharedMemory);
    return llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl

#endif //REMOVEGLOBALVARS_H
