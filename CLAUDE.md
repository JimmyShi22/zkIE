# 项目开发约定

## What this is

三段式可验证 ML 推理项目（commit / prove / verify 独立进程）。
实现细节以代码与 docs/ 为准，本文不做实现描述——设计演进时本节不改。

## 入口

全项目仅此五处，新增入口必须在此登记：

| 入口 | 位置 | 用途 |
|---|---|---|
| zkie-bench CLI | tools/zkie-bench/src/main.rs | 三阶段 commit/prove/verify + matrix/run |
| zkie-check CLI | tools/zkie-check/src/main.rs | 验收报告 |
| executor_smoke ×3 | models/{gpt2,gemma3,timesfm}/examples/executor_smoke.rs | 单进程冒烟（承诺→执行→验证） |

## 代码目录结构

```
crates/zkie-core/   基础件：field/fixed_point/codec、artifact/progress/error/types、
                     compute、storage、oracle、claim
crates/zkie-engine/ 执行环境：primitives、storage::mmap、compute::cpu、context、metrics
crates/zkie-ops/    算子与证明：op/registry、pipeline、proof/*、活算子模块
models/{gpt2,gemma3,timesfm}/  config + executor + executor_smoke（权重一律外置）
tools/              zkie-bench、zkie-check
scripts/            数据管线（提取/仿真/GT），非证明路径
docs/               spec / benchmarks / roadmap / overview（只描述当前实现）
```

## 构建与测试方法

- 构建：`cargo build --workspace` / `cargo check --workspace`；实测用 release
  （`cargo build --release -p zkie-bench`）。
- 测试：`cargo test --workspace` 全绿为准入门槛；单元测试写在模块内
  `#[cfg(test)]`，集成测试放 `tests/`；负例必须真实注入（序列化往返篡改后
  走完整验证路径断言拒绝），不允许只加不验。
- 三阶段实测：`zkie-bench matrix`（冷 1 + 暖 3）；数字同机实测、warm 中位数
  入账、冷跑不写台账。
- 打点（profile）方法：`zkie-bench profile-matrix --cases <case.toml>...
  --threads 1 8 64 --runs 3 --out <dir>`——阶段级（commit/prove/verify 墙钟与
  峰值内存）+ 算子级 + 内核级累计墙钟与调用次数；progress.jsonl 必须有真实
  elapsed 与已/总量；输出机器可读记录（每档线程×每阶段×每算子）+ 人读汇总
  （算子占比排序）；口径与 matrix 一致；三档线程都要出数，用于定位串行约束。
- 测试留痕（方案迭代树，强制，直接入库）：每次测试/基准运行，在仓库根目录
  `runs/` 下建目录：
  `<YYYYMMDD-HHMM>-<方案名>-<当前短commit>-<上一方案短commit>/`
  目录内必须包含并 commit 提交：
  - `NOTES.md`：方案名、基于哪次修改与哪个方案/基线（上一 commit）、运行
    目的、结论与未闭合项、关键数字（proof 字节 / 墙钟 / VmHWM / 测试通过数）；
  - 全部文本结果产物：result.json、progress.jsonl、profile.json、汇总文件、
    日志，一份不删。
  每跑一次建一个新目录（同一方案多轮 = 多个目录），形成可回溯的方案迭代树；
  超大二进制产物（如 proof.bin，数 GB）留在同目录磁盘上、gitignore 不入库，
  其余全部随方案 commit 提交。无出处的数字不得进入任何文档。

## 开发约定（硬规则）

1. 单实现原则：一个概念只允许一套实现；禁止旧实现/重复路径/legacy 模块残留。
2. 代码内不标注参照来源：代码、注释、文档、commit message 中一律不写任何
   外部来源线索（名称、缩写、路径、file:line、编号类标记）——让实现自足、
   让 AI 聚焦于代码本身；提交前自查。
3. commit：英文 message；每小步一 commit；TL 复核后才 push。
4. 资产红线：模型权重/GT/资产只读、外置，mmap/区间读取，禁止入库、禁止修改、
   禁止整张物化。
5. 测量纪律：优化必须有同机前后对照（proof 字节、墙钟、VmHWM）；字节不涨、
   VmHWM 不升、墙钟 3% 噪声内；无收益如实记录「未观察到收益」；不许改口径
   凑数字；所有结果留档并标明对应修改与方案。
6. 文档同步：删除功能时同步删除其文档描述；docs/ 与代码一致。

## 工作流红线

- 新机制必须从某个出处产生（有据可循），除非是老板或专家已经明确批准的方案。
- 不许自造等价机制过门禁。
- 技术拿不准报老板，不自行改方案。
