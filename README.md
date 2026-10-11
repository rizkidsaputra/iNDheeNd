# Konfigurasi pilot GNN4IFA

`pilot.yaml` menyimpan parameter eksperimen yang disepakati di `PROJECT_CONTEXT.md`.
`topology.txt` adalah representasi ringkas topologi dan endpoint untuk diterjemahkan ke
skenario ndnSIM. Berkas ini sengaja tidak menyimpan disk VM, hasil build, dataset besar,
atau kredensial.

Untuk fallback Mini-NDN, `minindn_topology.conf` memetakan topologi 10 router dan endpoint,
sedangkan `minindn_smoke.py` hanya memeriksa bahwa NFD dapat hidup pada topologi tersebut.
Smoke test bukan hasil eksperimen dan belum menjalankan serangan.

## Tahap 2 Mini-NDN

`minindn_stage2.py` adalah orchestrator awal untuk mengoperasikan topologi penelitian. Script ini:

- menyalakan NFD dan NLSR pada semua node;
- mengiklankan prefix producer dan prefix target unavailable-content;
- menjalankan `ndnpingserver` pada P1/P2;
- menjalankan consumer C1--C6;
- menjalankan attacker A1/A2 dengan `--attack-rate`;
- menjalankan collector pada M;
- menyimpan snapshot status tiap node dan statistik interface/link ke direktori log.

Jalankan mode non-interaktif selama 180 detik:

```bash
sudo python3 ~/GNN4IFA-config/minindn_stage2.py \
  ~/GNN4IFA-config/minindn_topology.conf \
  --duration 180 --attack-rate 100 --no-cli \
  --log-dir /tmp/gnn4ifa-stage2-rate100
```

Untuk masuk ke CLI Mini-NDN dan mengamati node selama proses berjalan, hilangkan `--no-cli`.
Uji rate 500 dan 1000 dilakukan sebagai run terpisah. Log per node berada di direktori utama,
log aplikasi memiliki suffix `-producer`, `-consumer`, atau `-attacker`, log collector berada
di `M-collector.log`, dan statistik per interface berada di subdirektori `links/`.

Script ini masih merupakan harness tahap 2. Sebelum memakai hasilnya untuk klaim penelitian,
verifikasi bahwa route NLSR sudah terbentuk, `ndnping` menerima Data dari P1/P2, rate aktual
attacker sesuai target, dan counter queue/PIT/NACK/timeout benar-benar terisi.

Parameter serangan adalah input pilot. Sebelum eksperimen utama, verifikasi tekanan PIT,
perilaku NACK/timeout, dampak pada trafik sah, dan apakah kehilangan laporan benar-benar
terjadi pada jalur pelaporan.
