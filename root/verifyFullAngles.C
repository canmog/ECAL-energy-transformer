// Independent post-update verifier for one full/MC/3dfit file triplet.
// It verifies entry/key alignment, every original 3dfit branch, mcEne, all angle
// branches, and the numerical direction convention for every entry.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <TBranch.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TObjArray.h>
#include <TTree.h>

void verifyFullAngles(const char* fullPath, const char* mcPath, const char* fitPath) {
  TFile full_file(fullPath, "READ");
  TFile mc_file(mcPath, "READ");
  TFile fit_file(fitPath, "READ");
  TTree* full = full_file.IsZombie() ? nullptr : static_cast<TTree*>(full_file.Get("tree3dfit"));
  TTree* mc = mc_file.IsZombie() ? nullptr : static_cast<TTree*>(mc_file.Get("t"));
  TTree* fit = fit_file.IsZombie() ? nullptr : static_cast<TTree*>(fit_file.Get("tree3dfit"));
  if (!full || !mc || !fit) {
    Printf("ANGLE_VERIFY_FAILED missing tree full=%d mc=%d fit=%d",
           full != nullptr, mc != nullptr, fit != nullptr);
    return;
  }
  const Long64_t n = full->GetEntries();
  if (mc->GetEntries() != n || fit->GetEntries() != n) {
    Printf("ANGLE_VERIFY_FAILED entry counts full=%lld mc=%lld fit=%lld",
           n, mc->GetEntries(), fit->GetEntries());
    return;
  }

  TObjArray* fit_branches = fit->GetListOfBranches();
  for (int i = 0; i < fit_branches->GetEntries(); ++i) {
    auto* source = static_cast<TBranch*>(fit_branches->At(i));
    TBranch* copied = full->GetBranch(source->GetName());
    if (!copied || std::string(copied->GetTitle()) != source->GetTitle()) {
      Printf("ANGLE_VERIFY_FAILED original branch mismatch name=%s", source->GetName());
      return;
    }
  }

  // Content-level preservation check: compare every leaf of every original
  // 3D-fit branch at deterministic positions across the file. Fixed-size cell
  // arrays are checked element by element. The all-entry key loop below provides
  // the exhaustive ordering check without rereading both 34 GB cell payloads.
  Long64_t sampled_values = 0, bad_sampled_values = 0;
  std::vector<Long64_t> sample_entries;
  for (int q = 0; q <= 10; ++q) sample_entries.push_back((n - 1) * q / 10);
  for (Long64_t entry : sample_entries) {
    full->GetEntry(entry);
    fit->GetEntry(entry);
    for (int i = 0; i < fit_branches->GetEntries(); ++i) {
      auto* source_branch = static_cast<TBranch*>(fit_branches->At(i));
      auto* copied_branch = full->GetBranch(source_branch->GetName());
      TObjArray* source_leaves = source_branch->GetListOfLeaves();
      TObjArray* copied_leaves = copied_branch->GetListOfLeaves();
      if (source_leaves->GetEntries() != copied_leaves->GetEntries()) {
        Printf("ANGLE_VERIFY_FAILED leaf-count mismatch branch=%s", source_branch->GetName());
        return;
      }
      for (int j = 0; j < source_leaves->GetEntries(); ++j) {
        auto* source_leaf = static_cast<TLeaf*>(source_leaves->At(j));
        auto* copied_leaf = static_cast<TLeaf*>(copied_leaves->At(j));
        if (source_leaf->GetLen() != copied_leaf->GetLen()) {
          Printf("ANGLE_VERIFY_FAILED leaf-length mismatch branch=%s entry=%lld",
                 source_branch->GetName(), entry);
          return;
        }
        for (int k = 0; k < source_leaf->GetLen(); ++k) {
          const double a = source_leaf->GetValue(k);
          const double b = copied_leaf->GetValue(k);
          ++sampled_values;
          if (!(a == b || (std::isnan(a) && std::isnan(b)))) ++bad_sampled_values;
        }
      }
    }
  }
  if (bad_sampled_values) {
    Printf("ANGLE_VERIFY_FAILED changed original data sampled_values=%lld bad=%lld",
           sampled_values, bad_sampled_values);
    return;
  }
  const char* required[] = {"mcEne", "mcTheta", "mcPhi", "mcDirX", "mcDirY",
                            "mcDirZ", "mcKX", "mcKY"};
  for (const char* name : required) {
    if (!full->GetBranch(name)) {
      Printf("ANGLE_VERIFY_FAILED missing merged branch %s", name);
      return;
    }
  }

  UInt_t fr = 0, fe = 0, mr = 0, me = 0, rr = 0, re = 0;
  Float_t mt = 0.f, mp = 0.f, mc_energy = 0.f, full_energy = 0.f;
  Float_t theta = 0.f, phi = 0.f, dx = 0.f, dy = 0.f, dz = 0.f, kx = 0.f, ky = 0.f;
  full->SetBranchStatus("*", 0);
  for (const char* name : {"_run", "_event", "mcEne", "mcTheta", "mcPhi", "mcDirX",
                           "mcDirY", "mcDirZ", "mcKX", "mcKY"})
    full->SetBranchStatus(name, 1);
  full->SetBranchAddress("_run", &fr); full->SetBranchAddress("_event", &fe);
  full->SetBranchAddress("mcTheta", &theta); full->SetBranchAddress("mcPhi", &phi);
  full->SetBranchAddress("mcDirX", &dx); full->SetBranchAddress("mcDirY", &dy);
  full->SetBranchAddress("mcDirZ", &dz); full->SetBranchAddress("mcKX", &kx);
  full->SetBranchAddress("mcKY", &ky);
  full->SetBranchAddress("mcEne", &full_energy);

  mc->SetBranchStatus("*", 0);
  for (const char* name : {"info.run", "info.event", "mcinfo.p", "mcinfo.theta", "mcinfo.phi"})
    mc->SetBranchStatus(name, 1);
  mc->SetBranchAddress("info.run", &mr); mc->SetBranchAddress("info.event", &me);
  mc->SetBranchAddress("mcinfo.p", &mc_energy);
  mc->SetBranchAddress("mcinfo.theta", &mt); mc->SetBranchAddress("mcinfo.phi", &mp);

  fit->SetBranchStatus("*", 0);
  fit->SetBranchStatus("_run", 1); fit->SetBranchStatus("_event", 1);
  fit->SetBranchAddress("_run", &rr); fit->SetBranchAddress("_event", &re);

  Long64_t bad_key = 0, bad_energy = 0, bad_value = 0, bad_norm = 0;
  double max_component_diff = 0.0;
  for (Long64_t i = 0; i < n; ++i) {
    full->GetEntry(i); mc->GetEntry(i); fit->GetEntry(i);
    if (fr != mr || fe != me || fr != rr || fe != re) ++bad_key;
    if (full_energy != mc_energy) ++bad_energy;
    const double ex = std::sin(mt) * std::cos(mp);
    const double ey = std::sin(mt) * std::sin(mp);
    const double ez = std::cos(mt);
    const double ekx = ex / ez;
    const double eky = ey / ez;
    const double diffs[] = {std::fabs(theta - mt), std::fabs(phi - mp),
                            std::fabs(dx - ex), std::fabs(dy - ey), std::fabs(dz - ez),
                            std::fabs(kx - ekx), std::fabs(ky - eky)};
    const double md = *std::max_element(diffs, diffs + 7);
    max_component_diff = std::max(max_component_diff, md);
    if (!std::isfinite(md) || md > 5.e-6) ++bad_value;
    if (std::fabs(dx * dx + dy * dy + dz * dz - 1.0) > 5.e-6) ++bad_norm;
  }
  if (bad_key || bad_energy || bad_value || bad_norm) {
    Printf("ANGLE_VERIFY_FAILED file=%s bad_key=%lld bad_energy=%lld bad_value=%lld "
           "bad_norm=%lld maxdiff=%g",
           fullPath, bad_key, bad_energy, bad_value, bad_norm, max_component_diff);
    return;
  }
  Printf("ANGLE_VERIFY_OK file=%s entries=%lld branches=%d original_fit_branches=%d "
         "sampled_original_values=%lld max_component_diff=%g",
         fullPath, n, full->GetNbranches(), fit->GetNbranches(), sampled_values,
         max_component_diff);
}
