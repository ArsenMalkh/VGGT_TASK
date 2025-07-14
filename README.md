VGGT_TASK
---

# Tennis Pointcloud Estimation

## 📦 Установка

1. **Клонируем VGGT и устанавливаем зависимости:**

```bash
git clone https://github.com/facebookresearch/vggt
cd vggt
pip install -r requirements.txt
```

2. **Скачиваем веса модели:**

Вес модели `vggt-1b.pt` можно взять из Hugging Face или установить локально.
Я скачал и сохранил вес на сервере вручную:

```
vggt-1b.pt
//https://disk.360.yandex.ru/d/3NI_QC5zIhp1tQ
```

## 🚀 Скрипты

### 1. `run_vggt.py`

Извлекает из видео RGB + высоту (в метрах) каждого пикселя, используя VGGT. На выходе — тензор `(T, 4, H, W)`, где каналы: R, G, B, высота.

**Пример запуска:**

```bash
python3 run_vggt.py \
    --video videos/video2.mp4 \
    --weights vggt-1b.pt \
    --device cuda \
    --resize 672
```

На выходе сохранится файл `rgbd_tensor.pt`.
