## 🤖 LSTM нейросеть, которая на основе начала фразы предсказывает её продолжение
Эта сеть:
- Обучается на задаче предсказания следующего токена по входной последовательности.
- Умеет генерировать текст (одно слово за другим) до окончания или достижения лимита длины.

Результаты обучения и тестирования:
<img src="https://github.com/StefanEpic/sprint2_LSTM/blob/main/training_plots.png" width="900" height="400" alt="График ROUGE">
```
avg_train_loss: 5.11
avg_rouge_1: 0.04
avg_rouge_2: 0.01
```
Пример генерации 1:
- Исходный текст: "this nice weather is making me really want to get my tattoo sleeve but gotta pay for the lsat"
- Переданный в обработку: "this nice weather is making me really want to get my tattoo sleeve but gotta"
- Результат обработки: "this nice weather is making me really want to get my tattoo sleeve but gotta get out of the same time i dont want to go"

Пример генерации 2:
- Исходный текст: "this nice weather is making me really want to get my tattoo sleeve but gotta pay for the lsat"
- Переданный в обработку: "this nice weather is making me really want to get my tattoo sleeve but gotta"
- Результат обработки: "this nice weather is making me really want to get my tattoo sleeve but gotta go to bed nows and tomorrow"

Пример генерации 3:
- Исходный текст: "this nice weather is making me really want to get my tattoo sleeve but gotta pay for the lsat"
- Переданный в обработку: "this nice weather is making me really want to get my tattoo sleeve but gotta"
- Результат обработки: "this nice weather is making me really want to get my tattoo sleeve but gotta go to bed i miss"

👉 [Описание основных функций для работы с моделью](https://github.com/StefanEpic/sprint2_LSTM/blob/main/solution.ipynb)
