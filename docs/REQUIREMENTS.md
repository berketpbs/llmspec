# llmspec — Proje Gereksinim Dokümanı

Kullanıcının donanımına (RAM, CPU, GPU/VRAM) göre hangi LLM modellerinin gerçekten iyi
çalışacağını tespit eden terminal aracı. Rust ile yazılır, Windows + Linux hedeflenir,
hem interaktif TUI hem klasik CLI modu sunar.

- **Proje / crate / binary adı:** `llmspec`
- **Ortam değişkeni öneki:** `LLMSPEC_`
- **Config dizini:** Linux `~/.config/llmspec/`, Windows `%APPDATA%\llmspec\`

---

## 1. Ürün Tanımı

Yüzlerce modeli kalite / hız / uyum / bağlam boyutlarında puanlayıp sıralar. Amaç
"bu modeli çalıştırabilir miyim?" sorusuna sayısal ve gerekçeli bir cevap vermek.

---

## 2. Kurulum Yöntemleri (gelecekte dağıtım için)

- Windows: `scoop install llmspec`
- macOS/Linux: Homebrew (`brew install llmspec`), MacPorts, `curl | sh` quick-install script
- `cargo install llmspec` / crates.io paketi
- `uv tool install llmspec` / `uvx` (Python wrapper — opsiyonel, öncelik değil)
- Docker/Podman image (JSON çıktı üretir, `jq` ile sorgulanabilir)
- Kaynaktan derleme: `cargo build --release`

---

## 3. Donanım Tespiti (`hardware` modülü)

- **RAM/CPU**: `sysinfo` crate ile toplam/kullanılabilir RAM, çekirdek sayısı
- **NVIDIA GPU**: `nvidia-smi` ile (Windows + Linux'ta mevcut). Çoklu-GPU desteği —
  VRAM'leri toplar. Komut başarısız olursa GPU model adından VRAM tahmini yap (fallback).
- **AMD GPU**: `rocm-smi` (Linux) — Windows'ta farklı bir yöntem gerekebilir
  (örn. WMI/DXGI sorgusu), bu ayrıca araştırılmalı.
- **Apple Silicon**: `system_profiler` — kapsam dışı (Windows/Linux hedefliyoruz) ama kod
  yapısında yer tutucu olarak bırakılabilir.
- **Intel Arc**: sysfs (`mem_info_vram_total`, discrete) / `lspci` (integrated) — Linux
- **Backend tespiti**: Sonuca göre otomatik olarak CUDA / ROCm / SYCL / CPU (ARM) /
  CPU (x86) etiketi atanır — hız tahmininde kullanılır.
- **Windows özel**: NVIDIA GPU tespiti `nvidia-smi` üzerinden (kurulu ise) çalışır;
  RAM/CPU tespiti sorunsuz.
- **Override flag'leri**: `--memory=32G`, `--ram=128G`, `--cpu-cores=16` — otomatik tespit
  başarısız olursa veya farklı hedef donanım simüle etmek için. Kabul edilen birimler:
  G/GB/GiB, M/MB/MiB, T/TB/TiB (case-insensitive). GPU tespit edilmediyse `--memory`
  sentetik bir GPU girdisi oluşturur.

---

## 4. Model Veritabanı

- Kaynak: HuggingFace REST API'sinden bir Python scraper (`scripts/scrape_hf_models.py`,
  stdlib-only, pip bağımlılığı yok) ile çekilir, `data/hf_models.json`'a yazılır ve
  **derleme zamanında** (`include_str!`) binary'e gömülür.
- Yüzlerce model / onlarca sağlayıcı: Meta Llama, Mistral, Qwen, Google Gemma, Microsoft
  Phi, DeepSeek, IBM Granite, Allen Institute OLMo, xAI Grok, Cohere, BigCode, 01.ai,
  Upstage, TII Falcon, Zhipu GLM, Moonshot Kimi, Baidu ERNIE vb.
- Kategoriler: genel amaçlı, coding (CodeLlama, StarCoder2, Qwen2.5/3-Coder),
  reasoning (DeepSeek-R1), multimodal/vision (Llama 3.2 Vision, Qwen2.5-VL), chat,
  enterprise, embedding (nomic-embed, bge).
- **MoE (Mixture-of-Experts) tespiti**: model config'inden (`num_local_experts`,
  `num_experts_per_tok`) veya bilinen mimari eşlemelerinden otomatik algılanır.
  Örnek: Mixtral 8x7B toplam 46.7B parametre ama token başına yalnızca ~12.9B aktif →
  VRAM ihtiyacı 23.9 GB yerine ~6.6 GB'a düşer (expert offloading ile).
- GGUF kaynak zenginleştirme: unsloth ve bartowski gibi sağlayıcılardan indirilebilir GGUF
  linkleri eklenir, 7 günlük TTL cache ile (`data/gguf_sources_cache.json`).
  `--no-gguf-sources` ile atlanabilir.
- Güncelleme: `make update-models` veya `./scripts/update_models.sh` — mevcut veriyi
  yedekler, JSON'u doğrular, binary'i yeniden derler.

---

## 5. Kuantizasyon ve Bellek Hesaplama

- Sabit kuantizasyon varsaymak yerine **dinamik seçim**: Q8_0 (en yüksek kalite) → Q2_K
  (en sıkıştırılmış) hiyerarşisinde yukarıdan aşağıya dener, mevcut belleğe sığan en
  yüksek kaliteliyi seçer.
- Tam bağlamda (context) hiçbir şey sığmıyorsa, **yarım bağlamda** tekrar dener.
- VRAM: GPU çıkarımı için birincil kısıt. Sistem RAM'i: CPU-only çalıştırma için yedek.

---

## 6. Çok Boyutlu Skorlama Sistemi

Her model 4 boyutta 0–100 arası puanlanır:

| Boyut | Ölçtüğü şey |
|---|---|
| **Quality** | Parametre sayısı, model ailesi itibarı, kuantizasyon kaybı, görev uyumu |
| **Speed** | Backend, parametre sayısı ve kuantizasyona göre tahmini tok/s |
| **Fit** | Bellek kullanım verimliliği (tatlı nokta: mevcut belleğin %50–80'i) |
| **Context** | Kullanım senaryosu hedefine göre bağlam penceresi kapasitesi |

- Boyutlar **ağırlıklı composite skor**a dönüştürülür. Ağırlıklar kullanım senaryosuna
  (General, Coding, Reasoning, Chat, Multimodal, Embedding) göre değişir.
  Örnek: Chat → Speed ağırlığı 0.35 (daha yüksek); Reasoning → Quality ağırlığı 0.55.
- Modeller composite skora göre sıralanır; çalıştırılamayan modeller ("Too Tight") her
  zaman en altta.

---

## 7. Hız Tahmini

- LLM çıkarımı **bellek bant genişliği sınırlıdır**: her token, model ağırlıklarının
  tamamının VRAM'den bir kez okunmasını gerektirir.
- GPU modeli tanınıyorsa gerçek bellek bant genişliği kullanılır:
  `(bandwidth_GB_s / model_size_GB) × efficiency_factor`
- Verimlilik faktörü (varsayılan `0.55`) ve mod-başına hız çarpanları **kullanıcı
  tarafından ayarlanabilir** olmalı (Advanced Config).
- ~80 GPU'yu kapsayan bir bant genişliği tablosu (NVIDIA tüketici+datacenter, AMD, Apple).
- Tanınmayan GPU'lar için backend-bazlı sabit katsayılar
  (fallback: `K / params_b × quant_speed_multiplier`):

| Backend | Sabit |
|---|---|
| CUDA | 220 |
| Metal | 160 |
| ROCm | 180 |
| SYCL | 100 |
| CPU (ARM) | 90 |
| CPU (x86) | 70 |
| NPU (Ascend) | 390 |

---

## 8. Uyum (Fit) Analizi

**Çalışma modları:**

- **GPU** — model VRAM'e sığıyor, hızlı çıkarım
- **MoE** — expert offloading ile, aktif expert'ler VRAM'de, pasifler RAM'de
- **CPU+GPU** — VRAM yetersiz, kısmi GPU offload ile RAM'e taşar
- **CPU** — GPU yok, model tamamen sistem RAM'ine yüklenir

**Uyum seviyeleri:**

- **Perfect** — GPU'da önerilen bellek karşılanmış, GPU gerektirir
- **Good** — rahat sığıyor; MoE offload veya CPU+GPU için ulaşılabilecek en iyi seviye
- **Marginal** — sıkışık sığma, veya CPU-only (CPU-only her zaman burada tavan yapar)
- **Too Tight** — hiçbir yerde (VRAM veya sistem RAM) yeterli alan yok

---

## 9. TUI (İnteraktif Terminal Arayüzü) — `llmspec` (argümansız çalıştırıldığında)

Üstte sistem özellikleri (CPU, RAM, GPU adı, VRAM, backend) gösterilir. Modeller composite
skora göre sıralı, kaydırılabilir tabloda listelenir. Her satır: skor, tahmini tok/s,
donanıma en uygun kuantizasyon, çalışma modu, bellek kullanımı, kullanım-senaryosu
kategorisi.

### Normal Mod Tuş Bağlamaları

| Tuş | Eylem |
|---|---|
| `Up`/`Down` veya `j`/`k` | Modeller arası gezinme |
| `/` | Arama modu (isim, sağlayıcı, parametre, kullanım senaryosu üzerinde kısmi eşleşme) |
| `Esc`/`Enter` | Arama modundan çık |
| `Ctrl-U` | Aramayı temizle |
| `f` | Uyum filtresini döngüle: All, Runnable, Perfect, Good, Marginal |
| `a` | Kullanılabilirlik filtresini döngüle: All, GGUF Avail, Installed |
| `s` | Sıralama sütununu döngüle: Score, Params, Mem%, Ctx, Date, Use Case |
| `v` | Visual moda gir (çoklu model seçimi) |
| `V` | Select moda gir (sütun-bazlı filtreleme) |
| `t` | Renk temasını döngüle (otomatik kaydedilir) |
| `p` | Seçili model için Plan modu (donanım planlama) |
| `P` | Sağlayıcı filtre popup'ı |
| `U` | Kullanım-senaryosu filtre popup'ı |
| `C` | Yetenek filtre popup'ı |
| `L` | Lisans filtre popup'ı |
| `R` | Runtime/backend filtre popup'ı (llama.cpp, MLX, vLLM) |
| `S` | Donanım simülasyon popup'ı (RAM/VRAM/CPU override) |
| `A` | Gelişmiş yapılandırma popup'ı (verimlilik, mod faktörleri ayarı) |
| `b` | Topluluk lider tablosu görünümü |
| `I` | Çıkarım bench görünümü (yerel modellere karşı kalite skorlama) |
| `h` | Yardım popup'ı (tüm tuş bağlamaları) |
| `m` | Seçili modeli karşılaştırma için işaretle |
| `c` | Karşılaştırma görünümünü aç (işaretli vs seçili) |
| `x` | Karşılaştırma işaretini temizle |
| `i` | Kurulu-önce sıralamayı aç/kapa |
| `d` | Seçili modeli indir (birden fazla sağlayıcı varsa seçim popup'ı) |
| `D` | Download Manager'ı aç |
| `r` | Yüklü modelleri runtime sağlayıcılardan yenile |
| `Enter` | Seçili model için detay görünümünü aç/kapa |
| `PgUp`/`PgDn` | 10'ar kaydır |
| `g`/`G` | Başa/sona git |
| `q` | Çık |

### Vim-tarzı Modlar

- **Normal mod**: varsayılan, yukarıdaki tüm tuşlar aktif
- **Visual mod (`v`)**: `v` ile ankor, `j`/`k` ile ardışık satır aralığı seçimi.
  `c` çoklu-karşılaştırma görünümü açar, `m` iki-model karşılaştırma için işaretler,
  `Esc`/`v` çıkış. Çoklu-karşılaştırma tablosunda satırlar = öznitelik (Score, tok/s, Fit,
  Mem%, Params, Mode, Context, Quant), sütunlar = modeller; en iyi değerler vurgulanır;
  `h`/`l` yatay kaydırma.
- **Select mod (`V`)**: `h`/`l` ile sütun başlıkları arası gezinme, `Enter`/`Space` o
  sütunun eylemini tetikler:

  | Sütun | Filtre eylemi |
  |---|---|
  | Inst | Kullanılabilirlik filtresini döngüle |
  | Model | Arama moduna gir |
  | Provider | Sağlayıcı popup'ı |
  | Params | Parametre-boyut aralığı popup'ı (<3B, 3-7B, 7-14B, 14-30B, 30-70B, 70B+) |
  | Score/tok/s/Mem%/Ctx/Date | O sütuna göre sırala |
  | Quant | Kuantizasyon popup'ı |
  | Mode | Çalışma-modu popup'ı (GPU, MoE, CPU+GPU, CPU) |
  | Fit | Uyum filtresini döngüle |
  | Use Case | Kullanım-senaryosu popup'ı |

  Satır navigasyonu Select modda da çalışır (`j`/`k`, ok tuşları, `Ctrl-U`, `Ctrl-D`,
  `PageUp`/`PageDown`, `Home`/`End`).

### Plan Modu (`p`)

Normal uyum analizinin tersi: "modelim donanıma sığar mı?" değil, "bu model
konfigürasyonu için ne kadar donanım gerekir?"

| Tuş | Eylem |
|---|---|
| `Tab`/`j`/`k` | Düzenlenebilir alanlar arası geçiş (Context, Quant, Target TPS) |
| `Left`/`Right` | Alan içinde imleç hareketi |
| Yazı yazma | Aktif alanı düzenle |
| `Backspace`/`Delete` | Karakter sil |
| `Ctrl-U` | Alanı temizle |
| `Esc`/`q` | Plan modundan çık |

Gösterilenler: minimum/önerilen VRAM-RAM-CPU çekirdeği, uygulanabilir çalışma yolları
(GPU, CPU offload, CPU-only), daha iyi uyum hedefine ulaşmak için yükseltme farkları.

### Donanım Simülasyonu (`S`)

RAM/VRAM/CPU çekirdek sayısını override ederek farklı hedef donanımlarda hangi modellerin
sığacağını gösterir. Tüm skorlar/uyum seviyeleri/hız tahminleri anında yeniden hesaplanır.

| Tuş | Eylem |
|---|---|
| `Tab`/`j`/`k` | RAM/VRAM/CPU alanları arası geçiş |
| Rakam yazma | Seçili alanı düzenle |
| `Enter` | Simülasyonu uygula |
| `Ctrl-R` | Gerçek tespit edilen donanıma sıfırla |
| `Esc` | İptal ve kapat |

Simülasyon aktifken sistem çubuğunda ve durum çubuğunda `SIM` rozeti görünür.

### Gelişmiş Yapılandırma (`A`)

Hız/skor hesaplama parametrelerini ayarlar (tok/s tahmini bazı modellerde abartılı
çıkabiliyor). Değişiklikler anında uygulanır.

| Alan | Açıklama | Varsayılan |
|---|---|---|
| Efficiency | Bant genişliği bazlı TPS için global verimlilik faktörü | 0.55 |
| GPU factor | Salt GPU çıkarımı için hız çarpanı | 1.0 |
| CPU Offload | Ağırlıklar sistem RAM'ine taştığında hız çarpanı | 0.5 |
| MoE Offload | MoE expert değişimi için hız çarpanı | 0.8 |
| Tensor Par | Tensor-paralel çıkarım için hız çarpanı | 0.9 |
| CPU Only | Salt CPU çalıştırma için hız çarpanı | 0.3 |
| Context cap | Bellek tahmini için maks. bağlam uzunluğu | auto |

Tuşlar: `Tab`/`j`/`k` (alan geçişi), rakam/`.` yazma, `Left`/`Right`,
`Backspace`/`Delete`, `Ctrl-U` (temizle), `Enter` (uygula), `Esc`/`q` (uygulamadan kapat).

### Download Manager (`D`)

Tam ekran görünüm, üç bölüm:

- **Active Download** — ilerleme çubuğu, model adı, durum mesajı
- **Config** — GGUF model dizini (düzenlenebilir, kalıcı)
- **History** — geçmiş indirmeler (en yeni önce): model adı, sağlayıcı, durum, tarih.
  Başarısız indirmeler geçmişten kaldırılabilir; başarılı indirmeler sağlayıcıdan
  silinebilir.

| Tuş | Eylem |
|---|---|
| `Tab`/`Shift-Tab` | Odak döngüsü: Active → Config → History |
| `j`/`k`/oklar | Geçmiş listesinde gezin |
| `x` | Seçili modeli sil (onay ister) |
| `y`/`n` | Silmeyi onayla/iptal et |
| `e` | İndirme dizinini düzenle |
| `Enter` | Dizin düzenlemesini onayla |
| `Esc`/`D`/`q` | Kapat, model tablosuna dön |

Başarısız indirmelerde (örn. 404) `x` geçmişten kaldırır. Başarılı indirmelerde model
sağlayıcıdan silinir (Ollama ve llama.cpp için desteklenir).

### Topluluk Lider Tablosu (`b`)

Teorik hız tahmini yerine **gerçek dünya performans verisi** — aynı donanıma sahip diğer
kullanıcılardan ölçülen tok/s, TTFT (time-to-first-token), tepe VRAM kullanımı. Harici bir
topluluk benchmark veritabanı gerektirir (kendi backend'imizi kurmamız gerekir ya da bu
özellik ilk sürümde atlanır — büyük bağımlılık).

Sütunlar: Model (HF ID), Engine (llama.cpp/vLLM/Ollama/MLX), Quant, tok/s, Total t/s,
TTFT, VRAM, Ctx, User (doğrulanmış kullanıcılar `*` ile işaretli).

| Tuş | Eylem |
|---|---|
| `j`/`k`/oklar | Sonuçlarda gezin |
| `H` | Donanım seçici aç (herhangi bir GPU'yu tara) |
| `r` | API'den yenile |
| `b`/`q`/`Esc` | Kapat |

`H` ile 27 popüler GPU/chip'ten (RTX 5090'dan CPU-only'e, Apple M1-M4, AMD RX/MI, NVIDIA
datacenter) birini seçip o donanımın benchmark'larını yükleyebilme. "My Hardware
(auto-detect)" ile kendi sistemine dönme.

**API key kurulumu**: Genel benchmark'lar auth gerektirmez. Tam erişim için ortam değişkeni
(`LLMSPEC_API_KEY`) veya CLI flag ile API key sağlanır.

### Çıkarım Bench (`I`)

Yerel çalışan sağlayıcılara (Ollama, vLLM, MLX) karşı **canlı çıkarım benchmark'ı**
çalıştırır — gerçek istek gönderip TTFT, TPS, toplam gecikmeyi ölçer. Topluluk lider
tablosundan farkı: kendi gerçek donanımın + kendi gerçek modellerin ölçülür.

| Tuş | Eylem |
|---|---|
| `I` | Bench'i aç (sağlayıcıyı otomatik tespit edip çalıştırır) |
| `I` (tekrar) | Bench görünümü içinden yeniden çalıştır |
| `j`/`k`/oklar | Model sonuçlarında gezin |
| `Enter` | Seçili model için detay görünümü |
| `r` | Yönlendirme matrisi görünümüne geç |
| `q`/`Esc` | Bench görünümünü kapat |

Sonuçlar `~/.config/llmspec/bench-cache.json`'a önbelleklenir, sonraki açılışlarda anında
yüklenir.

CLI eşdeğerleri:

```
llmspec bench                                   # otomatik tespit + benchmark
llmspec bench --all                             # tüm keşfedilen modeller
llmspec bench --provider ollama llama3.2        # belirli model
llmspec bench --provider ollama --url http://my-server:11434 llama3.2
llmspec bench --json                            # scripting için JSON
llmspec bench --quality                         # rol-bazlı kalite skorlama
llmspec bench --quality --routing               # yönlendirme matrisi
```

Ortam değişkenleri: `OLLAMA_HOST` (varsayılan `http://localhost:11434`),
`VLLM_PORT` (varsayılan `8000`).

### Temalar (`t`)

10 yerleşik renk teması arasında döngü. Seçim `~/.config/llmspec/theme`'e otomatik
kaydedilir, sonraki açılışta yüklenir. Örnekler: Default, Dracula, Solarized, Nord,
Monokai, Gruvbox, Catppuccin (Latte/Frappé/Macchiato/Mocha).

### Web Dashboard

JSON-olmayan modda çalıştırıldığında otomatik olarak arka planda `0.0.0.0:8787` üzerinde
bir web dashboard başlar; aynı ağdaki herhangi bir tarayıcıdan `http://<makine-ip>:8787`
ile açılabilir.

Ortam değişkenleri: `LLMSPEC_DASHBOARD_HOST` (varsayılan `0.0.0.0`),
`LLMSPEC_DASHBOARD_PORT` (varsayılan `8787`). `--no-dashboard` ile devre dışı bırakılabilir.

---

## 10. CLI Modu

`--cli` veya herhangi bir subcommand ile klasik tablo çıktısı:

```
llmspec --cli                                   # tüm modeller, uyuma göre sıralı
llmspec fit --perfect -n 5                      # sadece mükemmel uyanlar, top 5
llmspec system                                  # tespit edilen sistem özellikleri
llmspec list                                    # veritabanındaki tüm modeller
llmspec search "llama 8b"                       # isim/sağlayıcı/boyuta göre arama
llmspec info "Mistral-7B"                       # tek modelin detaylı görünümü
llmspec recommend --json --limit 5              # top 5 öneri (JSON, agent/script için)
llmspec recommend --json --use-case coding --limit 3
llmspec recommend --force-runtime llamacpp      # otomatik MLX seçimini bypass et
llmspec plan "Qwen/Qwen3-4B" --context 8192
llmspec plan "..." --context 8192 --quant q4_k_m
llmspec plan "..." --context 8192 --target-tps 25 --json
llmspec serve --host 0.0.0.0 --port 8787        # REST API modu
```

### JSON Çıktı

`--json` bayrağı herhangi bir subcommand'a eklenebilir (makine-okunabilir çıktı için).
`recommend` için JSON varsayılan formattır.
`plan` JSON'ı içerir: request (`context`, `quantization`, `target_tps`), tahmini
min/önerilen donanım, yol-bazlı uygulanabilirlik (`gpu`, `cpu_offload`, `cpu_only`),
yükseltme farkları.

### Bağlam-Uzunluğu Sınırlama

`--max-context` — bellek tahmininde kullanılan bağlam uzunluğunu sınırlar (modelin reklamı
yapılan maks. bağlamını değiştirmeden):

```
llmspec --max-context 4096 --cli
llmspec --max-context 8192 fit --perfect -n 5
```

Belirtilmezse `OLLAMA_CONTEXT_LENGTH` ortam değişkeni (varsa) kullanılır.

---

## 11. REST API (`llmspec serve`)

TUI/CLI ile aynı fit/skorlama verisini HTTP üzerinden sunar, cluster scheduler/agregatörler
için:

```
GET  /health                                    # liveness
GET  /api/v1/system                             # düğüm donanım bilgisi
GET  /api/v1/models?min_fit=marginal&runtime=llamacpp&sort=score&limit=20
GET  /api/v1/models/top?limit=5&min_fit=good&use_case=coding
GET  /api/v1/models/{search}?runtime=any        # isim/sağlayıcı metin arama
```

Desteklenen query parametreleri: `limit`/`n`, `perfect` (true/false),
`min_fit` (perfect|good|marginal|too_tight), `runtime` (any|mlx|llamacpp),
`use_case` (general|coding|reasoning|chat|multimodal|embedding),
`provider` (alt-metin filtre), `search` (serbest metin),
`sort` (score|tps|params|mem|ctx|date|use_case),
`include_too_tight` (varsayılan `/top`'ta false, `/models`'ta true),
`max_context`, `force_runtime` (mlx|llamacpp|vllm).

---

## 12. Runtime Sağlayıcı Entegrasyonları

Birden fazla uyumlu sağlayıcı varsa, TUI'de `d` tuşu bir sağlayıcı seçim popup'ı açar.

### Ollama

- Gereksinim: Ollama kurulu ve çalışıyor olmalı (`ollama serve` veya masaüstü uygulama)
- Varsayılan endpoint: `http://localhost:11434`, otomatik tespit edilir
- Uzak Ollama: `OLLAMA_HOST` ortam değişkeni ile (`http://192.168.1.100:11434` gibi)
- İşleyiş: başlangıçta `GET /api/tags` ile kurulu modeller listelenir (TUI'de yeşil ✓ ile
  işaretlenir); `d` tuşu `POST /api/pull` tetikler, satırda animasyonlu ilerleme göstergesi
- Model adı eşleme: HF isimleri (`Qwen/Qwen2.5-Coder-14B-Instruct`) ↔ Ollama isimleri
  (`qwen2.5-coder:14b`) arasında **tam/kesin** eşleme tablosu tutulmalı (yaklaşık eşleşme
  yanlış modele yönlendirebilir)

### llama.cpp

- Gereksinim: `llama-cli`/`llama-server` PATH'te olmalı; HuggingFace'e ağ erişimi
- İşleyiş: HF modelleri bilinen GGUF repo'larına eşler (heuristic fallback ile), GGUF
  dosyalarını yerel önbelleğe indirir, eşleşen dosyalar varsa "kurulu" işaretler
- Ortam değişkenleri: `LLAMA_CPP_PATH` (binary dizini, PATH'ten önce kontrol edilir),
  `LLAMA_SERVER_PORT` (varsayılan `8080`)

### Docker Model Runner

- Gereksinim: Docker Desktop + Model Runner etkin; varsayılan endpoint
  `http://localhost:12434`
- İşleyiş: `GET /engines` ile model listesi; Ollama-tarzı tag eşleme (`ai/<tag>` formatı);
  `d` tuşu `docker model pull` çalıştırır
- Uzak: `DOCKER_MODEL_RUNNER_HOST` ortam değişkeni

### LM Studio

- Gereksinim: LM Studio çalışıyor, yerel sunucusu etkin; varsayılan endpoint
  `http://127.0.0.1:1234`
- İşleyiş: `GET /v1/models` ile liste; `d` tuşu `POST /api/v1/models/download` tetikler;
  ilerleme `GET /api/v1/models/download-status` polling ile takip edilir; HF isimleri
  doğrudan kabul edilir (eşleme gerekmez)
- Uzak: `LMSTUDIO_HOST` ortam değişkeni

### MLX (Apple Silicon — kapsamımızda opsiyonel/düşük öncelik)

MLX indirmeleri orijinal model yayıncısı yerine `mlx-community/*` HF repo'larına eşlenir.

---

## 13. Platform Desteği

| Platform | Durum | Yöntem |
|---|---|---|
| **Windows** | Tam destek hedefi | RAM/CPU tespiti native; NVIDIA GPU `nvidia-smi` (kurulu ise) |
| **Linux** | Tam destek hedefi | GPU tespiti: `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), sysfs/`lspci` (Intel Arc), `npu-smi` (Ascend) |

GPU tespit tablosu:

| Sağlayıcı | Tespit yöntemi | VRAM raporlama |
|---|---|---|
| NVIDIA | `nvidia-smi` | Kesin dedicated VRAM |
| AMD | `rocm-smi` (Linux) / Windows'ta ayrı çözüm gerekli | Tespit edilir (VRAM bilinmeyebilir) |
| Intel Arc (discrete) | sysfs | Kesin dedicated VRAM |
| Intel Arc (integrated) | `lspci` | Paylaşımlı sistem belleği |

Otomatik tespit başarısız olursa veya yanlış değer raporlarsa `--memory`/`--ram`/
`--cpu-cores` ile override.

---

## 14. Proje Yapısı

```
src/
  main.rs         -- CLI argüman ayrıştırma, giriş noktası, TUI başlatma
  hardware.rs     -- Sistem RAM/CPU/GPU tespiti (çoklu-GPU, backend belirleme)
  models.rs       -- Model veritabanı, kuantizasyon hiyerarşisi, dinamik quant seçimi
  fit.rs          -- Çok boyutlu skorlama (Q/S/F/C), hız tahmini, MoE offloading
  providers.rs    -- Runtime sağlayıcı entegrasyonu (Ollama, llama.cpp, Docker Model
                     Runner, LM Studio), kurulum tespiti, pull/indirme
  display.rs      -- Klasik CLI tablo render + JSON çıktı
  tui_app.rs      -- TUI uygulama durumu, filtreler, navigasyon
  tui_ui.rs       -- TUI render (ratatui)
  tui_events.rs   -- TUI klavye olay işleme (crossterm)
data/
  hf_models.json  -- Model veritabanı (derleme zamanında gömülü)
scripts/
  scrape_hf_models.py       -- HuggingFace API scraper
  update_models.sh          -- Otomatik veritabanı güncelleme
```

---

## 15. Bağımlılıklar (Rust crate'leri)

| Crate | Amaç |
|---|---|
| `clap` | CLI argüman ayrıştırma (derive makrolarıyla) |
| `sysinfo` | Çapraz-platform RAM/CPU tespiti |
| `serde`/`serde_json` | Model veritabanı JSON (de)serileştirme |
| `tabled` | CLI tablo formatlama |
| `colored` | CLI renkli çıktı |
| `ureq` | Runtime/sağlayıcı API entegrasyonu için HTTP istemcisi |
| `ratatui` | Terminal UI framework |
| `crossterm` | ratatui için terminal input/output backend |

---

## 16. Model Ekleme Süreci

1. Modelin HF repo ID'sini scraper'ın hedef listesine ekle
2. Model gated ise (HF auth gerektiriyorsa) parametre sayısı + context uzunluğuyla fallback
   girdisi ekle
3. Otomatik güncelleme scriptini çalıştır
4. Güncellenmiş model listesini doğrula
5. Dokümantasyonu güncelle

---

## 17. Kapsam Dışı Bırakılabilecekler (v1 için önceliksiz)

- Apple Silicon / MLX desteği (Windows+Linux hedefliyoruz)
- Topluluk lider tablosu (harici bir benchmark servisi/backend gerektirir)
- Agent-framework skill entegrasyonu (isteğe bağlı ileri aşama)
- Docker Model Runner, LM Studio entegrasyonları — Ollama + llama.cpp'den sonra eklenebilir

---

## 18. v1 Geliştirme Sırası

1. Donanım tespiti (Windows + Linux, NVIDIA + RAM/CPU önce, AMD sonra)
2. Küçük elle-yazılmış model veritabanı (10-20 model) → sonra HF scraper
3. Kuantizasyon + bellek hesaplama + fit/skorlama mantığı
4. Klasik CLI tablo çıktısı (`--cli`, `list`, `search`, `info`, `recommend`, `system`)
5. TUI iskeleti: sistem bilgisi üstte, model tablosu, temel navigasyon
   (`j`/`k`, `/` arama, `f` filtre, `q` çık)
6. TUI genişletme: Plan modu, Hardware Simulation, temalar
7. Ollama entegrasyonu (kurulu model tespiti + indirme)
8. REST API (`serve` komutu)
9. Download Manager, Advanced Config, llama.cpp entegrasyonu
10. (Opsiyonel) Inference Bench, Community Leaderboard, diğer sağlayıcılar
