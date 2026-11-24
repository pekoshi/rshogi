# p2shogi の python スクリプト

p2shogi の評価関数を学習・変換・検証するときに使ったスクリプトです。手順は [pekoshi/p2shogi](https://github.com/pekoshi/p2shogi) の `docs/TRAINING_FROM_SCRATCH.md` と `docs/REPRODUCE.md` を見てください。

- フォルダ名は開発時の実験フォルダの名前のままです。スクリプトどうしの import は相対パスに直してあります。
- データ、exe、net の絶対パス (`E:/shogi/...` など) は開発環境の値です。各自の環境に合わせて書き換えてください。
- 対局の器の一部 (`v4confirm/v4c.py` など) は、比較相手のエンジンのパスや option を含みます。他のエンジンの評価関数や実行ファイルは同梱していません。
