# 🏓 Paddle Game with Enhanced AI

Bu proje, derin Q-öğrenme (Deep Q-Learning) kullanarak paddle oyunu için gelişmiş bir AI sistemi içerir.

## 🚀 Özellikler

### 🤖 AI Modelleri
- **DQN AI** (`dqn_ai.py`): Derin Q-Network ile gelişmiş AI
- **Simple AI** (`paddle_ai.py`): Basit Q-Learning tabanlı AI

### 🧠 Gelişmiş AI Özellikleri

#### 1. **Enhanced State Vector**
- **Önceki**: 4 boyutlu state (ball_y, paddle_y, ball_direction_x, ball_direction_y)
- **Yeni**: 6 boyutlu state (ball_y, paddle_y, ball_direction_x, ball_direction_y, **ball_velocity_x**, **paddle_height**)
- **Normalizasyon**: Tüm değerler ekran boyutlarına göre normalize edilir

#### 2. **Double DQN Implementation**
```python
# Main network ile en iyi aksiyonu seç
best_actions = torch.argmax(self.model(next_states), dim=1)
# Target network ile Q değerini hesapla
next_q_values = self.target_model(next_states).gather(1, best_actions.unsqueeze(1))
```

#### 3. **Target Network Updates**
- Her 100 adımda bir target network güncellenir
- Daha stabil eğitim sağlar

#### 4. **Logarithmic Epsilon Decay**
```python
self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(-decay_rate * self.steps)
```

#### 5. **Robust Loss Calculation**
```python
target_tensor = current_q_values.clone().detach()
loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
```

#### 6. **Debugging ve Monitoring**
- Her fonksiyona docstring eklendi
- Action seçimlerinde print/log çıktıları
- NaN/Inf kontrolü için assert'ler
- Training progress monitoring

## 📁 Dosya Yapısı

```
PaddleGame/
├── dqn_ai.py              # Gelişmiş DQN AI
├── paddle_ai.py           # Basit Q-Learning AI
├── paddle_game.py         # Ana oyun dosyası
├── ai_test_untrained.py   # Eğitilmemiş model testi
└── README.md              # Bu dosya
```

## 🎮 Kullanım

### Eğitim
```bash
cd PaddleGame
python paddle_game.py train
```

### Oyun Oynama
```bash
cd PaddleGame
python paddle_game.py
```

### Eğitilmemiş Model Testi
```bash
cd PaddleGame
python ai_test_untrained.py
```

## 🔧 Teknik Detaylar

### State Vector (6 boyutlu)
1. `ball_y / WINDOW_HEIGHT` - Topun Y pozisyonu (normalize)
2. `paddle_y / WINDOW_HEIGHT` - Paddle'ın Y pozisyonu (normalize)
3. `ball_direction_x / BALL_SPEED_X` - Topun X yönü (normalize)
4. `ball_direction_y / BALL_SPEED_Y` - Topun Y yönü (normalize)
5. `ball_velocity_x / 5.0` - Topun X hızı (normalize) ⭐ **YENİ**
6. `paddle_height / PADDLE_HEIGHT` - Paddle yüksekliği (normalize) ⭐ **YENİ**

### Reward Sistemi
- **Top vurma**: +2 puan
- **Merkeze yakın vurma**: +1 ek puan
- **Top kaçırma**: -2 puan
- **Topa yakın olma**: +0.2 puan

### Eğitim Parametreleri
- **Learning Rate**: 0.001
- **Gamma (Discount Factor)**: 0.99
- **Epsilon Decay**: Logaritmik
- **Target Network Update**: Her 100 adım
- **Batch Size**: 32
- **Memory Size**: 10,000

## 🧪 Test Senaryoları

### 1. Eğitilmemiş Model Testi
`ai_test_untrained.py` dosyası ile eğitilmemiş modelin davranışını gözlemleyebilirsiniz:
- Rastgele hareketler
- Yüksek epsilon değeri
- Frame sayısı ve epsilon gösterimi

### 2. Eğitim Öncesi/Sonrası Karşılaştırma
1. `ai_test_untrained.py` çalıştır (eğitilmemiş)
2. `paddle_game.py train` ile eğitim yap
3. `paddle_game.py` ile eğitilmiş modeli test et

## 📊 Monitoring ve Debugging

### Console Çıktıları
```
DQN AI initialized with state_size=6, action_size=3
Random action: 1 (epsilon: 0.950)
Greedy action: 0 (Q-values: [0.1, 0.3, 0.2])
Training step 1000, Loss: 0.0456, Epsilon: 0.368
Target network updated at step 100
```

### Debug Assertions
- State değerlerinde NaN/Inf kontrolü
- Reward tipi kontrolü
- Done değeri boolean kontrolü

## 🎯 Performans İyileştirmeleri

1. **CPU Optimizasyonu**: GPU yerine CPU kullanımı
2. **Küçük Network**: 64-32-3 mimarisi
3. **Efficient Memory**: 10,000 deneyim limiti
4. **Frequent Updates**: Her 100 adımda target network güncelleme

## 🔮 Gelecek Geliştirmeler

- [ ] Prioritized Experience Replay
- [ ] Dueling DQN
- [ ] Multi-agent training
- [ ] Visual state representation
- [ ] Curriculum learning

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir. 