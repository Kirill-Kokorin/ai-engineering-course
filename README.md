# ai-engineering-course
##TROUBLESHOOTING
- Invalid credentials format. Please use only base64 credentials (Authorization data, not client secret!)
Что видел?
ошибка после ломания ключа GigaChat
Почему?
нарушилась строка Base64
Что сделал?
исправил ключ, в будущем благодаря перенаправлению на провайдера HF эта ошибка не появлялась

- 'ascii' codec can't encode characters
Что видел?
ошибка после ломания ключа GigaChat
Почему?
в ключ GigaChat добавил русские буквы
Что сделал?
исправил ключ