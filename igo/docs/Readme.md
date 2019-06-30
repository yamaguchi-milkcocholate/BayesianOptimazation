
# Sphinxでのドキュメント作成（メモ）
* <http://docs.sphinx-users.jp/tutorial.html>

## 準備
* コメントをdocstringで書いておく
* <http://docs.sphinx-users.jp/domains.html#the-python-domain>
* <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>

## 設定（Makefileなどができる，設定済み）
```shell
# Sphinx のインストール
pip install Sphinx

# 雛形の生成（いくつかの質問に答える）
# プロジェクト名，作者，バージョンなど．Sphinx extensionsは基本Yesで
sphinx-quickstart
```

## 使い方
```shell
# .rstファイルの作成
# sphinx-apidoc [options] -o outputdir packagedir [pathnames]
sphinx-apidoc -o . ..

# ビルド
make html
```

* conf.pyの変更
	+ sys.path.insert(0, os.path.abspath('..'))
	+ 'sphinx.ext.pngmath'をextensionsに追加
	+ html_theme = 'classic'に変更