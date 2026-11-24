//! bucket_jump_measure binary entry point。
//!
//! 詳細は [`tools::bucket_jump_measure_tool`] 参照。

fn main() -> anyhow::Result<()> {
    tools::bucket_jump_measure_tool::run()
}
