# gmx_harness

[English](README.md) | 日本語

GROMACS のシミュレーションパイプライン（EM → NVT → NPT → 本計算、溶媒和、box 操作、Martini、AWH、BAR）を、ファイルと bash スクリプトの一式として生成するライブラリです。AI エージェント（と人間）が安全に扱うためのハーネスも同梱しています。
mylibs（`yagaiG-libs`）の `gromacs` パッケージを切り出したものです。

- **何も実行しない**: ライブラリは `gmx` を呼びません。生成された `run.sh` は人またはジョブスケジューラが実行します。
  GROMACS・PLUMED には依存せず、依存は pip パッケージのみです（pydantic, numpy, OpenMM, MDAnalysis, pandas, openpyxl）。
- **自由度**: どの mdp オプションも `additional_mdp_parameters` で指定できます。`Calculation` を継承すれば独自ステップも作れます。
  任意のコマンドは `RawShellStep` で実行できます（明示的な許可が必要）。
- **安全性**:
  - `plan → preview → write` の流れ。書き込む前にすべての衝突を確認します
    （preview なしに書くのは、チェックの小さな記録ファイル `checks.record_facts` の `<file>.facts.json` と
    `Report.enforce(out_dir=...)` の `checks.json` だけです）
  - 設計チェック（`gmx_harness.checks`）が既定で走り、理由付きで waive されていないエラーがあると書き込みを拒否します
  - 上書き・削除するのは gmx_harness 自身が生成したファイル（`.gmx_harness_manifest.json` にハッシュを記録）だけで、人が編集したファイルは拒否します
  - 対話的な `input()` はどこにもありません（エージェントが止まりません）
  - ステップ名・ファイル名・define・残基名を検証し、スクリプトに入る値はすべて `shlex.quote` します
  - mdp の検証（置換し忘れのプレースホルダ、数値の範囲、グループ数、行の注入など）
  - 破壊的・未検査の操作には `allow_unsafe=True` / `confirm=True` / `force_modified=True` が必要です

## インストール

```bash
pip install "gmx-harness @ git+<このリポジトリの URL>"
```

## 使い方

```python
from gmx_harness import EM, MD, MDType, OverwritePolicy, build_plan, save_json

steps = [
    EM(),
    MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt", nsteps=50_000,
       useRestraint=True, defines=["POSRES"]),
    MD(type=MDType.v_rescale_c_rescale, calculation_name="npt", nsteps=500_000, gen_vel="no"),
]
save_json(steps, "pipeline.json")

plan = build_plan(steps, "start.gro", "work", extra_inputs=["topo.top", "mol.itp"])
# 名前を変えて置く / 途中のステップにファイルを置く:
# build_plan(..., extra_inputs={"topo.top": "MOL_fixed.top"}, step_inputs={"npt": {"index.ndx": "MOL.ndx"}})
print(plan.summary())
print(plan.preview())
plan.write()                                     # OverwritePolicy.ERROR（既定）
# plan.write(OverwritePolicy.REPLACE_GENERATED)  # パラメータを変えて作り直すとき
```

生成されるディレクトリ:

```
work/
  run.sh                     # 全ステップを順に実行し、最初の失敗で止まる
  .gmx_harness_manifest.json
  0_em/   setting.mdp grommp.sh mdrun.sh run.sh copy.sh input.gro topo.top mol.itp
  1_nvt/  ...
  2_npt/  ...
```

実行: `cd work && bash run.sh`。ステップのディレクトリに `freeze` ファイルを置くと、そのステップは止まります。

再実行・続きからの実行:
- `run.sh` を再実行すると、終わったステップは飛ばします。終わったステップが、同じく終わっている後ろのステップの入力を上書きすることはありません（再投入やコピー・clone したツリーでも安全）。
- 中断した MD ステップは、grompp をやり直さずに `output.cpt` から続けます。
- 延長: `cd work/6_md_prod && bash extend.sh 200000`（ps 単位、`gmx convert-tpr -extend` を使用）の後、もう一度 `bash run.sh`。
- `generate_xtc.sh [グループ] [出力名]`。例: `bash generate_xtc.sh MOL mol_whole.xtc`。
- ステップ間で引き継ぐファイル: `build_plan(..., carry=("*.top", "*.itp", "*.ndx"))`（既定は top と itp）。
- PLUMED: `MD(..., plumed=text)` で plumed.dat を置き、`-plumed plumed.dat` で実行します。チェックポイントから続けるときは `RESTART` を自動で付けます（HILLS/COLVAR は追記されます）。
- `output.gro` を作らずに終わったステップがあると、トップの `run.sh` はエラーで止まります。

スクリプトには生成したマシン固有の情報を一切書き込まず、GROMACS は実行時に探します。そのためディレクトリをそのままクラスタへコピーできます。実行するマシンで使える環境変数:

| 変数 | 意味 | 既定値 |
|---|---|---|
| `GMX` | GROMACS のコマンド | PATH 上の `gmx_d`、`gmx_mpi`、`gmx` の最初に見つかったもの |
| `MDRUN_ARGS` | すべての `mdrun` に追加する引数（例: `"-ntomp 8 -gpu_id 0"`） | なし |

GROMACS 以外にスクリプトが必要とするのは bash と awk だけです（溶媒和後の topology 編集は awk で行います）。
`require_preflight=True` のときは `sha256sum`（GNU coreutils）も必要です。
実行前に、いつもの方法で GROMACS を読み込んでください（`module load gromacs`、`source .../GMXRC`、ジョブスクリプトなど）。

### ステップ

| クラス | 用途 |
|---|---|
| `EM`, `MartiniEM` | エネルギー最小化 |
| `MD(type=MDType.*)`, `MartiniMD` | MD（NVT / NPT / NH+PR） |
| `AWH`, `BarMethod` | 自由エネルギー |
| `Solvation`, `SolvationMCH`, `SolvationSCP216`, `RuntimeSolvation`（非推奨） | 溶媒和 |
| `RemoveResidue`, `ResizeBox`, `AddFiles` | 型付きのファイル操作 |
| `RawShellStep`（`allow_unsafe=True` が必要） | 任意の bash |

そのほか: `MDParameters`（mdp の編集と検証）、`GroFile`（.gro/.ndx/.xyz/.pdb）、`load_xvg`、
`generate_inermolecular_interactions`、`gmx_harness.relax.relax`（OpenMM）、`gmx_harness.analysis`（MDAnalysis）。

### index ファイル・topology の下準備・バッチジョブ

```python
from gmx_harness import molecule_atoms, write_ndx, prepare_topology, write_job_scripts

fiber = molecule_atoms(169, range(216))                      # 分子 0..215 の原子番号（1 始まり）
write_ndx("MOL_fiber.ndx", {"Fiber1": fiber, "fiberA": fiber})
prepare_topology("MOL.top", "MOL_fixed.top", nmols=216, itp_name="MOL_hbond.itp")   # 分子数 + #ifdef INTER の include
write_job_scripts("calc", ["MOL_fiber_rot_+10"], open("Gromacs.sbatch").read())    # {JOB_NAME}/{SCRIPT} を含むテンプレート
```

`write_job_scripts` は、自分のジョブテンプレートを各系のディレクトリに書き出し、`submit.sh` / `submit_restart.sh`（`sbatch -d singleton`）も作ります。ファイルを書くだけで投入はせず、編集済みのファイルは上書きしません。

### テンプレートから PLUMED 入力を作る

```python
from gmx_harness import MD, MDType, Layout, MoleculeLabels, preprocess_file

layout = Layout(MoleculeLabels.from_gro("MOL_labeled.gro"), nmol=60, nros=6)  # 残基名の列 = 断片ラベル
text = preprocess_file("metad_twist.plumed.in", layout, {"NDISK": 10, "BIAS_IFACE": 4})
metad = MD(type=MDType.v_rescale_c_rescale, calculation_name="metad", gen_vel="no", plumed=text)
```

テンプレートは PLUMED の入力に `#define` / `#for v in a..b` / `#endfor` / `#include`、`{式}`、
`@sel(disk=, mol=, res=, name=, heavy=)`（選んだ断片の原子範囲）を加えたものです。式はサンドボックス内で評価され
（`__` で始まる名前・lambda・import は不可）、`#include` はテンプレートのディレクトリの外を読めません。

### 設計チェック

`build_plan` は既定で `gmx_harness.checks` を実行します（純 Python で、何も実行しません）。input.gro と topo.top の対応、
maxwarn、`strict_mdp=False`、生のシェルステップ、PLUMED の入力と stride などを調べます。問題にはそれぞれ固定のコード
（`L003`、`P002` など）が付きます。エラーがあると `preview().ok` が False になり、`write()` は拒否します。
ただし理由を書いて waive したものは除きます:

```python
plan = build_plan(steps, "start.gro", "work", extra_inputs=["topo.top"],
                  waive={"L004": "grompp warning about ... is expected here"})
```

waiver には既知のコードと空でない理由が必要です。`expand_template` は展開後の PLUMED テンプレートを調べ
（テンプレートが使わない define も報告）、`check_tree` は実行後のログと COLVAR を読みます。
`require_preflight=True` にすると、`preflight.ok`（ワークスペース側の preflight スクリプトが grompp/plumed を実行して書く）が
存在して内容が一致するまで run.sh は起動を拒否します。コードの一覧: `gmx_harness.checks.CODES`
（`harness/llm_docs/api.md` にも記載）。

### .gro から構造を組み立てる

```python
from gmx_harness import Assembly, GroFile, make_oligorosette, make_rosette2, precoordinate2

mono = precoordinate2(GroFile.from_gro_file("MOL.gro"), 1, 2, 3)   # 原子番号: 頂点, NH, O
ring = make_rosette2(mono, n=6, size=0.35)                          # 6 分子の Assembly
for i, m in enumerate(ring):
    m.set_residue_number(i + 1)
ring_gro = ring.to_gro(renumber=True)
fiber = make_oligorosette(ring_gro, n=10, length=0.35, angle=10.0)  # リングを積み重ねる（slip= でらせん）
fiber.to_gro(box=(10, 10, 3.5), renumber=True).save_gro("fiber.gro")
```

ほかに `pre_coordinate`、`make_rosette`、`make_half_rosette2`、`Assembly.translate/rotate`、
`gmx_harness.build.rotation` / `align`（3x3 行列。`GroFile.rotate` は scipy の `Rotation` も受け付けます）があります。
mylibs の `pre_coordinator` / `rosette_maker` からの移植で、座標は 1e-16 nm 程度で一致します。

## mylibs からの移行

| mylibs | gmx_harness |
|---|---|
| `gromacs.calculation.*` | `gmx_harness.*`（クラス名は同じ） |
| `gromacs.mdp` | `gmx_harness.mdp`（`check(key)` はそのまま使え、`validate()` / `ensure_valid()` を追加） |
| `gromacs.itp`, `gromacs.relax`, `gromacs.analyzing` | `gmx_harness.itp`, `gmx_harness.relax`, `gmx_harness.analysis` |
| `gromacs.pre_coordinator`, `gromacs.rosette_maker` | `gmx_harness.build`（`precoordinate2`, `make_rosette2`, `make_oligorosette`, `Assembly` など） |
| `mole.gro.GroFile` | `gmx_harness.GroFile`（`mole` に依存しない） |
| `base_utils.plotlib.load_xvgdata` | `gmx_harness.load_xvg`（`XvgData` を返す。先頭行を捨てなくなった） |
| `launch(..., full_overwrite)` | `confirm=True` が必要で、gmx_harness が作ったディレクトリにだけ使える |
| `save_json` / `load_json`（`__class__` 付き辞書のリスト） | 独自のバージョン付き形式 `{"format": "gmx_harness.pipeline", "version": 1, "steps": [{"type", "params"}]}`。mylibs の JSON は読めない |
| `FileControl(name, cmd)` | `RawShellStep(name, cmd, allow_unsafe=True)`。`FileControl.remove_MCH` → `RemoveResidue` |
| `sed -i '/MCH/d' topol.top` | `RemoveResidue` / `ResizeBox`: 生成スクリプト内の awk で、[ molecules ] と #include だけを編集（topol.top の綴り違いも解消） |

意図的に変えた挙動: 各ステップのスクリプトは `set -eo pipefail` で失敗時に止まる。`freeze` は終了コード 1 で止まる。
`generate_xtc.sh` は対話しない・ovito を起動しない。`print` の出力は `logging`（`gmx_harness` ロガー）に出る。
`run.sh` の再実行は終わったステップの入力を上書きせず、チェックポイントがあれば grompp を飛ばす。

## 開発

```bash
uv sync
uv run python -m unittest discover -s tests -t .
uv run mypy
uv run python -c "from gmx_harness.apidoc import write_api_docs; write_api_docs()"   # docstring を変えたら
```
