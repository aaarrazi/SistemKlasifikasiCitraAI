# Fixtures pengujian — BUKAN data training

- `fake_class_indices.json` — mapping label **sintetis** untuk pengujian urutan
  indeks. **Jangan** dipakai di produksi: urutan label asli harus berasal dari
  skrip training Anda (`json.dump(train_generator.class_indices, ...)`)
  dan disimpan sebagai `class_indices.json` di root proyek.
- Gambar contoh dibuat saat runtime oleh `tests/conftest.py`
  (`make_image_bytes`) agar repo tidak menyimpan file biner.
