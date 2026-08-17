// Add MC incidence-direction truth to one existing merged ECAL ROOT file.
//
// Safety contract:
//   * the input full file is opened read-only;
//   * every existing branch is cloned to a temporary file in the same directory;
//   * seven new Float_t branches are appended;
//   * entry count, original branch names/titles, and new branches are verified;
//   * the original pathname is replaced atomically only after every check passes.
//
// Matching follows ../merge/mergeMcEneBatch.C: (info.run, info.event) in the MC
// tree "t" is matched to (_run, _event) in "tree3dfit". No entry-order
// assumption is used while writing.
//
// New branches (angles in radians):
//   mcTheta, mcPhi              spherical MC direction
//   mcDirX, mcDirY, mcDirZ      incoming unit-vector components
//   mcKX, mcKY                  dx/dz, dy/dz slopes

#include <cmath>
#include <cerrno>
#include <cstdio>
#include <map>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <TBranch.h>
#include <TFile.h>
#include <TObjArray.h>
#include <TSystem.h>
#include <TTree.h>

namespace {

using RunEvt = std::pair<UInt_t, UInt_t>;

struct AngleTruth {
  Float_t theta;
  Float_t phi;
  Float_t dir_x;
  Float_t dir_y;
  Float_t dir_z;
  Float_t kx;
  Float_t ky;
};

const char* kAngleBranches[] = {
    "mcTheta", "mcPhi", "mcDirX", "mcDirY", "mcDirZ", "mcKX", "mcKY"};

bool finiteTruth(const AngleTruth& a) {
  return std::isfinite(a.theta) && std::isfinite(a.phi) &&
         std::isfinite(a.dir_x) && std::isfinite(a.dir_y) &&
         std::isfinite(a.dir_z) && std::isfinite(a.kx) && std::isfinite(a.ky);
}

bool sameTruth(const AngleTruth& a, const AngleTruth& b) {
  const float eps = 1.e-6f;
  return std::fabs(a.theta - b.theta) < eps && std::fabs(a.phi - b.phi) < eps &&
         std::fabs(a.dir_x - b.dir_x) < eps && std::fabs(a.dir_y - b.dir_y) < eps &&
         std::fabs(a.dir_z - b.dir_z) < eps && std::fabs(a.kx - b.kx) < eps &&
         std::fabs(a.ky - b.ky) < eps;
}

std::vector<std::pair<std::string, std::string>> branchSchema(TTree* tree) {
  std::vector<std::pair<std::string, std::string>> out;
  TObjArray* branches = tree->GetListOfBranches();
  for (int i = 0; i < branches->GetEntries(); ++i) {
    auto* branch = static_cast<TBranch*>(branches->At(i));
    out.emplace_back(branch->GetName(), branch->GetTitle());
  }
  return out;
}

bool containsAllOriginal(
    TTree* tree,
    const std::vector<std::pair<std::string, std::string>>& original,
    std::string& why) {
  for (const auto& item : original) {
    TBranch* branch = tree->GetBranch(item.first.c_str());
    if (!branch) {
      why = "missing original branch " + item.first;
      return false;
    }
    if (item.second != branch->GetTitle()) {
      why = "changed title for original branch " + item.first + ": '" +
            item.second + "' -> '" + branch->GetTitle() + "'";
      return false;
    }
  }
  return true;
}

}  // namespace


void addMcAngleBatch(const char* fullPath, const char* mcPath, bool commit = true) {
  const std::string full_path(fullPath);
  const std::string mc_path(mcPath);
  const std::string tmp_path = full_path + ".angle-tmp.root";

  // Idempotence: a fully updated file is left untouched. A partial schema is a
  // hard failure because silently mixing versions would be worse than stopping.
  {
    TFile check(fullPath, "READ");
    TTree* tree = check.IsZombie() ? nullptr : static_cast<TTree*>(check.Get("tree3dfit"));
    if (!tree) {
      Printf("ANGLE_UPDATE_FAILED cannot open tree3dfit in %s", fullPath);
      return;
    }
    int present = 0;
    for (const char* name : kAngleBranches) present += tree->GetBranch(name) != nullptr;
    if (present == static_cast<int>(sizeof(kAngleBranches) / sizeof(kAngleBranches[0]))) {
      Printf("ANGLE_ALREADY_PRESENT file=%s entries=%lld branches=%d",
             fullPath, tree->GetEntries(), tree->GetNbranches());
      return;
    }
    if (present != 0) {
      Printf("ANGLE_UPDATE_FAILED partial angle schema (%d/7 branches) in %s",
             present, fullPath);
      return;
    }
  }

  // Build the truth map using only four small scalar branches from the large MC file.
  TFile mc_file(mcPath, "READ");
  TTree* mc = mc_file.IsZombie() ? nullptr : static_cast<TTree*>(mc_file.Get("t"));
  if (!mc) {
    Printf("ANGLE_UPDATE_FAILED cannot open MC tree t in %s", mcPath);
    return;
  }
  UInt_t mc_run = 0, mc_event = 0;
  Float_t theta = 0.f, phi = 0.f;
  mc->SetBranchStatus("*", 0);
  mc->SetBranchStatus("info.run", 1);
  mc->SetBranchStatus("info.event", 1);
  mc->SetBranchStatus("mcinfo.theta", 1);
  mc->SetBranchStatus("mcinfo.phi", 1);
  mc->SetBranchAddress("info.run", &mc_run);
  mc->SetBranchAddress("info.event", &mc_event);
  mc->SetBranchAddress("mcinfo.theta", &theta);
  mc->SetBranchAddress("mcinfo.phi", &phi);

  std::map<RunEvt, AngleTruth> truth;
  Long64_t duplicate_same = 0, duplicate_conflict = 0, invalid_mc = 0;
  const Long64_t n_mc = mc->GetEntries();
  for (Long64_t i = 0; i < n_mc; ++i) {
    mc->GetEntry(i);
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double cp = std::cos(phi);
    const double sp = std::sin(phi);
    AngleTruth a;
    a.theta = theta;
    a.phi = phi;
    a.dir_x = static_cast<Float_t>(st * cp);
    a.dir_y = static_cast<Float_t>(st * sp);
    a.dir_z = static_cast<Float_t>(ct);
    if (std::fabs(ct) < 1.e-8) {
      a.kx = a.ky = std::numeric_limits<Float_t>::quiet_NaN();
    } else {
      a.kx = static_cast<Float_t>(a.dir_x / a.dir_z);
      a.ky = static_cast<Float_t>(a.dir_y / a.dir_z);
    }
    if (!finiteTruth(a)) {
      ++invalid_mc;
      continue;
    }
    const RunEvt key(mc_run, mc_event);
    auto inserted = truth.emplace(key, a);
    if (!inserted.second) {
      if (sameTruth(inserted.first->second, a)) ++duplicate_same;
      else ++duplicate_conflict;
    }
  }
  if (invalid_mc || duplicate_conflict) {
    Printf("ANGLE_UPDATE_FAILED MC invalid=%lld conflicting_duplicates=%lld file=%s",
           invalid_mc, duplicate_conflict, mcPath);
    return;
  }

  TFile input(fullPath, "READ");
  TTree* in = input.IsZombie() ? nullptr : static_cast<TTree*>(input.Get("tree3dfit"));
  if (!in) {
    Printf("ANGLE_UPDATE_FAILED cannot open input tree3dfit in %s", fullPath);
    return;
  }
  const Long64_t n_in = in->GetEntries();
  const auto original_schema = branchSchema(in);
  const int original_branch_count = in->GetNbranches();
  UInt_t run = 0, event = 0;
  in->SetBranchAddress("_run", &run);
  in->SetBranchAddress("_event", &event);

  if (!gSystem->AccessPathName(tmp_path.c_str())) gSystem->Unlink(tmp_path.c_str());
  TFile output(tmp_path.c_str(), "RECREATE");
  if (output.IsZombie()) {
    Printf("ANGLE_UPDATE_FAILED cannot create temporary file %s", tmp_path.c_str());
    return;
  }
  output.SetCompressionSettings(input.GetCompressionSettings());
  TTree* out = in->CloneTree(0);
  if (!out) {
    Printf("ANGLE_UPDATE_FAILED CloneTree(0) failed for %s", fullPath);
    output.Close();
    gSystem->Unlink(tmp_path.c_str());
    return;
  }

  AngleTruth a{};
  out->Branch("mcTheta", &a.theta, "mcTheta/F");
  out->Branch("mcPhi", &a.phi, "mcPhi/F");
  out->Branch("mcDirX", &a.dir_x, "mcDirX/F");
  out->Branch("mcDirY", &a.dir_y, "mcDirY/F");
  out->Branch("mcDirZ", &a.dir_z, "mcDirZ/F");
  out->Branch("mcKX", &a.kx, "mcKX/F");
  out->Branch("mcKY", &a.ky, "mcKY/F");

  Long64_t matched = 0, missed = 0;
  for (Long64_t i = 0; i < n_in; ++i) {
    in->GetEntry(i);
    const auto it = truth.find(RunEvt(run, event));
    if (it == truth.end()) {
      ++missed;
      a.theta = a.phi = a.dir_x = a.dir_y = a.dir_z = a.kx = a.ky =
          std::numeric_limits<Float_t>::quiet_NaN();
    } else {
      ++matched;
      a = it->second;
    }
    out->Fill();
  }

  output.cd();
  out->Write("", TObject::kOverwrite);
  output.Close();
  input.Close();
  mc_file.Close();

  if (missed != 0 || matched != n_in) {
    Printf("ANGLE_UPDATE_FAILED match check entries=%lld matched=%lld missed=%lld file=%s",
           n_in, matched, missed, fullPath);
    gSystem->Unlink(tmp_path.c_str());
    return;
  }

  // Reopen the temporary file and prove the old schema survived before replacement.
  {
    TFile verify(tmp_path.c_str(), "READ");
    TTree* tree = verify.IsZombie() ? nullptr : static_cast<TTree*>(verify.Get("tree3dfit"));
    std::string why;
    if (!tree || tree->GetEntries() != n_in ||
        !containsAllOriginal(tree, original_schema, why)) {
      Printf("ANGLE_UPDATE_FAILED temporary verification: %s entries_expected=%lld file=%s",
             why.c_str(), n_in, tmp_path.c_str());
      verify.Close();
      gSystem->Unlink(tmp_path.c_str());
      return;
    }
    for (const char* name : kAngleBranches) {
      if (!tree->GetBranch(name)) {
        Printf("ANGLE_UPDATE_FAILED missing new branch %s in temporary file", name);
        verify.Close();
        gSystem->Unlink(tmp_path.c_str());
        return;
      }
    }
  }

  if (!commit) {
    gSystem->Unlink(tmp_path.c_str());
    Printf("ANGLE_DRYRUN_OK file=%s mc=%s entries=%lld branches=%d->%d "
           "matched=%lld duplicate_same=%lld",
           fullPath, mcPath, n_in, original_branch_count, original_branch_count + 7,
           matched, duplicate_same);
    return;
  }

  // Same-directory POSIX rename is atomic. If it fails, the original remains intact.
  if (std::rename(tmp_path.c_str(), full_path.c_str()) != 0) {
    Printf("ANGLE_UPDATE_FAILED atomic rename errno=%d tmp=%s target=%s",
           errno, tmp_path.c_str(), fullPath);
    gSystem->Unlink(tmp_path.c_str());
    return;
  }

  Printf("ANGLE_UPDATE_OK file=%s mc=%s entries=%lld branches=%d->%d "
         "matched=%lld duplicate_same=%lld",
         fullPath, mcPath, n_in, original_branch_count, original_branch_count + 7,
         matched, duplicate_same);
}
