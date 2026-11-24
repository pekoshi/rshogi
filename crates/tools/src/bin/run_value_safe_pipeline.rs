//! Value-safe pipeline orchestrator
//!
//! Phase A → B/C → D → E を順次呼び出して、 model selection report を集約する。

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "run_value_safe_pipeline")]
struct Args {
    #[arg(long)]
    config: PathBuf,

    /// 実行する phase: all | a | bc | d | e
    #[arg(long, default_value = "all")]
    phase: String,

    /// 出力ディレクトリ override (config の training.output_dir に優先)
    #[arg(long)]
    out_dir: Option<PathBuf>,

    /// resume 元 .p2nn を override
    #[arg(long)]
    resume_model: Option<PathBuf>,

    /// 互換性: 今回未使用 (PSV→JSONL ingest は build_dataset 側で完了済み)
    #[arg(long)]
    hao_dir: Option<PathBuf>,

    /// 互換性: 今回未使用
    #[arg(long)]
    extra_dir: Option<PathBuf>,

    /// 試走用 max_records override
    #[arg(long)]
    max_records: Option<usize>,

    /// epochs override
    #[arg(long)]
    epochs: Option<u32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.hao_dir.is_some() || args.extra_dir.is_some() {
        eprintln!(
            "[note] --hao-dir / --extra-dir は今回未使用 (data ingest は build_dataset 側で完了済み)"
        );
    }

    let phase = args.phase.to_lowercase();
    let phases: Vec<&str> = match phase.as_str() {
        "all" => vec!["a", "bc", "d", "e"],
        "a" | "bc" | "d" | "e" => vec![phase.as_str()],
        other => bail!("不明な phase: {} ('all', 'a', 'bc', 'd', 'e')", other),
    };

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("target/release"));

    let cfg_path = fs::canonicalize(&args.config).unwrap_or_else(|_| args.config.clone());

    // out_dir 決定
    let cfg_text = fs::read_to_string(&cfg_path)?;
    let cfg_val: toml::Value = toml::from_str(&cfg_text)?;

    let out_dir = if let Some(d) = &args.out_dir {
        d.clone()
    } else {
        let s = cfg_val
            .get("training")
            .and_then(|t| t.get("output_dir"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("output_dir が config にない"))?;
        PathBuf::from(s)
    };
    fs::create_dir_all(&out_dir)?;

    let phase_a_dir = out_dir.join("phase_a");
    let phase_d_dir = out_dir.join("phase_d");
    let phase_e_dir = out_dir.join("phase_e");

    let mut report_lines: Vec<String> = Vec::new();
    report_lines.push("# Value-Safe Pipeline Selection Report\n\n".to_string());
    report_lines.push(format!("- Config: `{}`\n", cfg_path.display()));
    report_lines.push(format!("- Output: `{}`\n", out_dir.display()));
    report_lines.push(format!("- Phases: {:?}\n\n", phases));

    // ================================================================
    // Phase A
    // ================================================================
    if phases.contains(&"a") {
        println!("\n=== Phase A: check_value_targets ===");
        let bin = exe_dir.join("check_value_targets.exe");
        let train_jsonl = cfg_val
            .get("data")
            .and_then(|d| d.get("train_jsonl"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("data.train_jsonl missing"))?
            .to_string();
        let cfg_max = cfg_val
            .get("data")
            .and_then(|d| d.get("max_train_records"))
            .and_then(|v| v.as_integer())
            .unwrap_or(0) as usize;
        let max_records = args.max_records.unwrap_or(cfg_max);
        let cp_candidates = cfg_val
            .get("value_target")
            .and_then(|v| v.get("cp_scale_candidates"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_float())
                    .map(|f| format!("{}", f))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_else(|| "600,800,1000,1200".to_string());

        let status = Command::new(&bin)
            .arg("--input")
            .arg(&train_jsonl)
            .arg("--output")
            .arg(&phase_a_dir)
            .arg("--max-records")
            .arg(format!("{}", max_records))
            .arg("--cp-scale-candidates")
            .arg(&cp_candidates)
            .status()
            .context("check_value_targets 実行失敗")?;
        if !status.success() {
            bail!("Phase A failed (exit {})", status.code().unwrap_or(-1));
        }
        report_lines.push("## Phase A: PASS\n".to_string());
        report_lines.push(format!("- Report: `{}/PHASE_A_REPORT.md`\n", phase_a_dir.display()));
    }

    // ================================================================
    // Phase B/C
    // ================================================================
    if phases.contains(&"bc") {
        println!("\n=== Phase B/C: train_dual_safe ===");
        let bin = exe_dir.join("train_dual_safe.exe");
        let mut cmd = Command::new(&bin);
        cmd.arg("--config").arg(&cfg_path).arg("--output-dir").arg(&out_dir);
        if let Some(m) = &args.resume_model {
            cmd.arg("--resume-p2nn").arg(m);
        }
        if let Some(e) = args.epochs {
            cmd.arg("--epochs").arg(format!("{}", e));
        }
        let status = cmd.status().context("train_dual_safe 実行失敗")?;
        if !status.success() {
            bail!("Phase B/C failed (exit {})", status.code().unwrap_or(-1));
        }
        report_lines.push("\n## Phase B/C: PASS\n".to_string());
        report_lines.push(format!("- Output: `{}/`\n", out_dir.display()));
        report_lines.push(format!("- See `{}/RUN_REPORT.md`\n", out_dir.display()));
    }

    let best_p2nn = out_dir.join("best_value.p2nn");

    // ================================================================
    // Phase D
    // ================================================================
    if phases.contains(&"d") {
        println!("\n=== Phase D: quant_parity_check ===");
        if !best_p2nn.exists() {
            report_lines.push("\n## Phase D: SKIP (best_value.p2nn 未生成)\n".to_string());
        } else {
            let mut latest_ckpt: Option<(u32, PathBuf)> = None;
            for entry in fs::read_dir(&out_dir)? {
                let entry = entry?;
                let name = entry.file_name();
                let name_s = name.to_string_lossy().to_string();
                if let Some(rest) = name_s.strip_prefix("checkpoint_epoch") {
                    if let Some(num) = rest.strip_suffix(".bin") {
                        if let Ok(n) = num.parse::<u32>() {
                            if latest_ckpt.as_ref().map(|(p, _)| n > *p).unwrap_or(true) {
                                latest_ckpt = Some((n, entry.path()));
                            }
                        }
                    }
                }
            }
            if let Some((_, ckpt)) = latest_ckpt {
                let bin = exe_dir.join("quant_parity_check.exe");
                let status = Command::new(&bin)
                    .arg("--config")
                    .arg(&cfg_path)
                    .arg("--checkpoint")
                    .arg(&ckpt)
                    .arg("--model")
                    .arg(&best_p2nn)
                    .arg("--output")
                    .arg(&phase_d_dir)
                    .status()?;
                if status.success() {
                    report_lines.push("\n## Phase D: PASS\n".to_string());
                } else {
                    report_lines.push("\n## Phase D: REJECT\n".to_string());
                }
                report_lines.push(format!(
                    "- Report: `{}/QUANT_PARITY_REPORT.md`\n",
                    phase_d_dir.display()
                ));
            } else {
                report_lines.push("\n## Phase D: SKIP (no checkpoint)\n".to_string());
            }
        }
    }

    // ================================================================
    // Phase E
    // ================================================================
    if phases.contains(&"e") {
        println!("\n=== Phase E: eval_sanity_check ===");
        if !best_p2nn.exists() {
            report_lines.push("\n## Phase E: SKIP (best_value.p2nn 未生成)\n".to_string());
        } else {
            let bin = exe_dir.join("eval_sanity_check.exe");
            let status = Command::new(&bin)
                .arg("--config")
                .arg(&cfg_path)
                .arg("--model")
                .arg(&best_p2nn)
                .arg("--output")
                .arg(&phase_e_dir)
                .status()?;
            if status.success() {
                report_lines.push("\n## Phase E: ADOPT\n".to_string());
            } else {
                report_lines.push("\n## Phase E: REJECT\n".to_string());
            }
            report_lines.push(format!("- Report: `{}/SANITY_REPORT.md`\n", phase_e_dir.display()));
        }
    }

    let select_path = out_dir.join("SELECTION_REPORT.md");
    let mut f = File::create(&select_path)?;
    f.write_all(report_lines.concat().as_bytes())?;
    println!("\nSELECTION_REPORT.md → {}", select_path.display());

    Ok(())
}
