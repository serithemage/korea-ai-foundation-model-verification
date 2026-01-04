# Save (Quick Commit & Push)

빠른 커밋과 푸시를 수행합니다. 인자로 커밋 메시지를 받습니다.

## Arguments

$ARGUMENTS - 커밋 메시지 (선택적). 없으면 자동 생성

## Instructions

1. `git status`로 변경사항 확인
2. 변경사항이 없으면 "변경사항이 없습니다" 출력 후 종료
3. 변경사항이 있으면:
   - `git add .`로 모든 파일 스테이징
   - $ARGUMENTS가 있으면 해당 메시지로 커밋
   - $ARGUMENTS가 없으면 변경 내용 분석 후 자동으로 커밋 메시지 생성
4. `git push`로 원격 저장소에 푸시
5. 푸시 실패 시 `git push -u origin {현재브랜치}`로 재시도

## Example Usage

```
/save                          # 자동 커밋 메시지
/save docs: README 업데이트    # 지정된 커밋 메시지
```
