# rshogi (p2shogi 版)

ラム将棋 ([SH11235/rshogi](https://github.com/SH11235/rshogi)) を fork し、将棋エンジン [p2shogi](https://github.com/pekoshi/p2shogi) の開発で使った改修を載せたブランチです。

- 学習器 [pekoshi/bullet-shogi](https://github.com/pekoshi/bullet-shogi) (`p2shogi` ブランチ) は、このリポジトリの `crates/rshogi-core` に依存します。両方を同じ親ディレクトリに置いてください。
- 対局器 (`tournament`)、CSA クライアントなどの開発用ツールも含みます。
- ライセンスは元のラム将棋に従います。
