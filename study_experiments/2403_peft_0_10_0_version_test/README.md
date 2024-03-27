# peft 0.10.0 version test
* 신규 기능 2가지 테스트
* multi_lora_test - 서로 다른 어댑터 한 forward/generate 호출로 모두 사용
* layer_replication_test - 레이어 복사해서 깊이 늘리기 테스트
	* test both decoder & encoder models
	* yanolja/EEVE-Korean-Instruct-2.8B-v1.0, klue/bert-base