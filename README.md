# ChatGPT_API_test
ChatGPT APIのAssistant機能を用いて、ゲーム内のキャラクターと自由に会話するシステムのプロトタイプ
![system_overview](README_ss01.png?raw=true)

利点
- 会話スレッドの管理をAPI側で行える（プロンプトに合わせて自動で最適なログ取得）

欠点
- APIが持つデフォルトのファイル検索機能だと、チャンクの大きさなどが自由に制御できないため、自分でデータファイルを管理する必要がある
- その結果、プロンプトにデータを含めて送信しなければならず、プロンプトが肥大化
- データのembedding化にもOpenAIのAPIを用いているため使用料がかさむ