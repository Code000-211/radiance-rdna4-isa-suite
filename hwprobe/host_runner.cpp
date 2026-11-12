// hwprobe host runner: launches every probe kernel many times and counts mismatching output words.
// UNTESTED ON HARDWARE: written without a GPU or ROCm; only built against hwprobe/mock (g++).
//
//   host_runner CASES_TSV ITERATIONS RESULTS_JSON [CASE_SUBSTRING]
//
// CASES_TSV comes from `python3 hwprobe/generate.py --data DIR` (one row per kernel: case, variant,
// kernel, code object, grid, block, input file, expected file, output words, fill byte, rule).
// Kernels have the signature (unsigned *out, const unsigned *in); the code object is a raw gfx1201
// ELF loaded with hipModuleLoad. Before every launch `out` is filled with the fill byte.
//
// Exit: 0 every fixed variant matched the host-computed result in every launch (hazard variants are
// informational), 1 some fixed variant mismatched (a harness bug, or for d16_other_half evidence
// against the linter's relaxation), 2 setup or HIP runtime error.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define HIP_OK(x)                                                                             \
  do {                                                                                        \
    hipError_t e_ = (x);                                                                      \
    if (e_ != hipSuccess) {                                                                   \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      std::exit(2);                                                                           \
    }                                                                                         \
  } while (0)

struct Row {
  std::string name, variant, kernel, hsaco, in, expect, rule;
  unsigned grid = 0, block = 0, out_words = 0, fill = 0;
};

struct Result {
  Row row;
  unsigned launches = 0, mismatching_launches = 0;
  unsigned long long mismatched_words = 0;
  bool has_first = false;
  unsigned first_launch = 0, first_index = 0, first_got = 0, first_want = 0;
};

static void die(const std::string &why) {
  std::fprintf(stderr, "host_runner: %s\n", why.c_str());
  std::exit(2);
}

static std::vector<uint32_t> read_words(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot read " + path);
  std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (bytes.size() % 4) die(path + ": size is not a multiple of 4");
  std::vector<uint32_t> w(bytes.size() / 4);
  for (size_t i = 0; i < w.size(); ++i)  // little-endian on disk, independent of the host
    w[i] = uint32_t((unsigned char)bytes[4 * i]) | uint32_t((unsigned char)bytes[4 * i + 1]) << 8 |
           uint32_t((unsigned char)bytes[4 * i + 2]) << 16 | uint32_t((unsigned char)bytes[4 * i + 3]) << 24;
  return w;
}

static std::vector<Row> read_cases(const std::string &path) {
  std::ifstream f(path);
  if (!f) die("cannot read " + path);
  std::vector<Row> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> c;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, '\t')) c.push_back(cell);
    if (c.size() != 11) die("bad row in " + path + ": " + line);
    Row r;
    r.name = c[0]; r.variant = c[1]; r.kernel = c[2]; r.hsaco = c[3];
    r.grid = unsigned(std::stoul(c[4])); r.block = unsigned(std::stoul(c[5]));
    r.in = c[6]; r.expect = c[7];
    r.out_words = unsigned(std::stoul(c[8])); r.fill = unsigned(std::stoul(c[9])); r.rule = c[10];
    rows.push_back(r);
  }
  return rows;
}

static std::string json_str(const std::string &s) {
  std::string o = "\"";
  for (char ch : s) {
    if (ch == '"' || ch == '\\') o += '\\';
    if ((unsigned char)ch < 0x20) { o += ' '; continue; }
    o += ch;
  }
  return o + "\"";
}

static const char *verdict(const Result &r) {
  if (r.row.variant == "fixed") return r.mismatched_words ? "FIXED_VARIANT_MISMATCH" : "ok";
  return r.mismatched_words ? "hazard_observed" : "not_observed";
}

int main(int argc, char **argv) {
  if (argc < 4) die("usage: host_runner CASES_TSV ITERATIONS RESULTS_JSON [CASE_SUBSTRING]");
  const unsigned iterations = unsigned(std::stoul(argv[2]));
  const std::string filter = argc > 4 ? argv[4] : "";
  std::vector<Row> rows = read_cases(argv[1]);

  hipDeviceProp_t prop;
  HIP_OK(hipGetDeviceProperties(&prop, 0));
  const std::string arch = prop.gcnArchName;
  if (arch.rfind("gfx1201", 0) != 0 && !std::getenv("HWPROBE_ANY_GPU"))
    die("device 0 is " + arch + ", not gfx1201 (set HWPROBE_ANY_GPU=1 to run anyway)");

  std::map<std::string, hipModule_t> modules;
  std::vector<Result> results;
  bool fixed_failed = false;
  for (const Row &row : rows) {
    if (!filter.empty() && row.name.find(filter) == std::string::npos) continue;
    if (!modules.count(row.hsaco)) {
      hipModule_t m;
      HIP_OK(hipModuleLoad(&m, row.hsaco.c_str()));
      modules[row.hsaco] = m;
    }
    hipFunction_t fn;
    HIP_OK(hipModuleGetFunction(&fn, modules[row.hsaco], row.kernel.c_str()));

    const std::vector<uint32_t> in = read_words(row.in), want = read_words(row.expect);
    if (want.size() != row.out_words) die(row.expect + ": expected " + std::to_string(row.out_words) + " words");
    if (in.size() != size_t(row.grid) * row.block) die(row.in + ": wrong number of input words");
    void *d_in = nullptr, *d_out = nullptr;
    HIP_OK(hipMalloc(&d_in, in.size() * 4));
    HIP_OK(hipMalloc(&d_out, want.size() * 4));
    HIP_OK(hipMemcpy(d_in, in.data(), in.size() * 4, hipMemcpyHostToDevice));
    std::vector<uint32_t> got(want.size());

    Result res;
    res.row = row;
    for (unsigned it = 0; it < iterations; ++it) {
      HIP_OK(hipMemset(d_out, int(row.fill), want.size() * 4));
      void *params[] = {&d_out, &d_in};
      HIP_OK(hipModuleLaunchKernel(fn, row.grid, 1, 1, row.block, 1, 1, 0, nullptr, params, nullptr));
      HIP_OK(hipDeviceSynchronize());
      HIP_OK(hipMemcpy(got.data(), d_out, got.size() * 4, hipMemcpyDeviceToHost));
      unsigned long long bad = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == want[i]) continue;
        if (!res.has_first) {
          res.has_first = true;
          res.first_launch = it; res.first_index = unsigned(i); res.first_got = got[i]; res.first_want = want[i];
        }
        ++bad;
      }
      res.launches++;
      res.mismatched_words += bad;
      res.mismatching_launches += bad != 0;
    }
    HIP_OK(hipFree(d_in));
    HIP_OK(hipFree(d_out));
    if (row.variant == "fixed" && res.mismatched_words) fixed_failed = true;
    std::printf("%-20s %-6s %-24s launches %u  mismatched words %llu  (%s)\n", row.name.c_str(), row.variant.c_str(),
                row.rule.c_str(), res.launches, res.mismatched_words, verdict(res));
    results.push_back(res);
  }
  for (auto &m : modules) HIP_OK(hipModuleUnload(m.second));

  std::ofstream o(argv[3]);
  if (!o) die(std::string("cannot write ") + argv[3]);
  o << "{\n  \"schema\": \"radisa-hwprobe-results-1\",\n  \"device\": " << json_str(prop.name)
    << ",\n  \"arch\": " << json_str(arch) << ",\n  \"iterations\": " << iterations << ",\n  \"results\": [";
  for (size_t i = 0; i < results.size(); ++i) {
    const Result &r = results[i];
    o << (i ? "," : "") << "\n    {\"case\": " << json_str(r.row.name) << ", \"variant\": " << json_str(r.row.variant)
      << ", \"kernel\": " << json_str(r.row.kernel) << ", \"expect_rule\": " << json_str(r.row.rule)
      << ", \"launches\": " << r.launches << ", \"words_per_launch\": " << r.row.out_words
      << ", \"mismatched_words\": " << r.mismatched_words << ", \"mismatching_launches\": " << r.mismatching_launches
      << ", \"verdict\": " << json_str(verdict(r)) << ", \"first_mismatch\": ";
    if (r.has_first)
      o << "{\"launch\": " << r.first_launch << ", \"index\": " << r.first_index << ", \"got\": " << r.first_got
        << ", \"want\": " << r.first_want << "}";
    else
      o << "null";
    o << "}";
  }
  o << "\n  ],\n  \"fixed_variants_ok\": " << (fixed_failed ? "false" : "true") << "\n}\n";
  return fixed_failed ? 1 : 0;
}
