## **1. Business Problem**
#
##### **Context**
#
California merupakan negara bagian yang paling padat penduduknya antara negara bagian amerika dan yang paling banyak dikunjungi wisatawan hingga pembisnis dari negara lain. Selain menjadi tempat wisata yang di idamkan banyak orang untuk dikunjungi, ternyata california juga menjadi tempat impian banyak orang untuk tinggal disana karena memiliki keindahan kota, pantai hingga gaya hidup yang mewah. Dari sudut pandang perusahaan yang bergerak sebagai agent properti properti yang ada di california ini menjadi sebuah peluang besar dan harus di manfaatkan untuk menjalankan bisnisnya. Hal ini didukung juga dengan populasi orang di california terus meningkat setiap tahunnya seperti yang di kutip dari berita media [Strong Towns](https://www.strongtowns.org/journal/2017/8/6/california-housing-crisis) hampir menyentuh angka 2% setiap tahunnya kenaikan populasi orang di california.
#
Walaupun bisnis properti memiliki peluang besar di california perlu juga menjadi pertimbangan dalam menentukan harga rumah, karena jika harga rumah yang terlalu tinggi akan mengakibatkan rumah tersebut tidak terjual dan ini berdampak pada lambatnya perputaran uang untuk agent properti dan yang paling berisiko adalah agent properti sebagai perantaran akan kehilangan client (pemilik rumah yang ingin menjual rumah). Namun jika harga yang dijual terlalu rendah / dibawah harga yang di inginkan client maka hal ini akan berisiko agent properti kehilangan client (pemilik rumah yang ingin menjual rumah) dan bisa berisiko client pindah ke agent properti lain.
#
Selama ini agent properti menentukan harga rumah dengan cara membandingkan properti yang ingin dijual dengan properti lain disekitar daerah tersebut dan terkadang harga yang dijual cenderung tinggi dan terkadang harga yang ditawarkan bisa saja rendah. Ketidakstabilan penentuan harga rumah ini sangat berisiko untuk agen properti seperti berita yang dikutip dari media [Strong Towns - News California Housing Crisis 1990 ](https://www.strongtowns.org/journal/2017/8/6/california-housing-crisis) , perubahan harga perumahan di california yang awalnya mengalami kenaikan secara drastis dari 5% hingga 25% pada tahun tahun 1985 sampai tahun 1989 namun mengalami penurunan drastis pada tahun 1990 dengan persentase penurunan dari 25% turun harga rumah -25%. Penurunan perubahan harga rumah ini di akibatkan karena harga perumahan di california pada tahun 1985 sampai tahun 1989 mengalami kenaikan harga yang sangat drastis mengakibatkan pembeli tidak mampu membeli rumah tersebut. Kenaikan harga rumah setinggi itu dikarenakan agent properti mengikuti harga pasar yang sedang naik tanpa mempertimbangkan spesifikasi rumah secara detail, karena kurang tepatnya penentuan harga rumah ini membuat penjualan rumah pada tahun 1990 mengalami penurunan bahkan harga perumahan juga ikut mengalami penurunan.

**Problem Statement**
#
Pengaruh dari kurang tepatnya dalam penentuan harga rumah ini berdampak pada penurunan penjualan agent properti yang menyebabkan harga tidak stabil dan customer tidak mampu membeli karena harganya yang naik secara drastis. Hal menjadi tantangan untuk agent properti untuk **menentukan harga rumah yang dijual secara tepat**, karena dalam menentukan harga rumah memiliki banyak faktor yang mendukung penentuan tinggi atau rendahnya harga rumah seperti lokasi rumah, umur rumah,jumlah ruangan,jumlah kamar tidur, keramaian lingkungan rumah, kawasan perumahan elit atau tidak dan jarak rumah dengan lokasi wisata. Tidak cukup hanya dengan membandingkan kondisi rumah yang akan dijual dengan rumah lainnya yang berada disekitar kemudian langsung menentukan harga rumah dengan harga pasar atau langsung menentukan hanya mengikuti harga pasar.

**Goals**
#
Berdasarkan permasalahan diatas, agen properti memerlukan **perangkat yang dapat membantu memperediksi harga rumah secara tepat**, perangkat ini disebut dengan machine learning. Machine learning ini dapat agen properti *gunakan ketika client (pihak pemilik rumah) ingin menjual rumahnya dan agen properti sebagai perantara* dapat menggunakan Machine learning ini untuk memprediksi harga rumah yang sesuai berdasarkan faktor-faktor lokasi rumah, umur rumah,jumlah ruangan,jumlah kamar tidur, keramaian lingkungan rumah, kawasan perumahan elit atau tidak dan jarak rumah dengan lokasi wisata dengan menggunakan machine learning.

**Analytic Approach**

Jadi, dalam hal ini yang perlu dilakukan : 
1. Menganalisis data untuk dapat menemukan pola dari fitur-fitur yang ada, yang membedakan satu perumahan dengan yang lainnya
2. Membangun model regresi yang dapat membantu agent properti untuk menyediakan sebuah perangkat prediksi harga rumah
3. Menemukan fitur yang memiliki pengaruh besar terhadap target **[median_house_values]** agar model yang dibangun lebih sederhana dan memiliki tingkat akurat yang baik

**Metric Evaluation**
#
Evaluasi matrik pertama yang akan digunakan adalah Mean Absolute Error (MAE). Mean Absolute Error adalah rataan nilai absolut dari error yang dihasilkan oleh model regresi. Jika nilai yang di hasilkan dari evaluasi metrik MAE 0 (nol) artinya akurasi dari hasil prediksi harga rumah mendekati akurat dan jika nilai dari MAE 1 (satu) maka akurasi dari prediksi dari hasil prediksi harga rumah tidak mendekati nilai yang akurat.
#
Penggunaan evaluasi matrik **MAE untuk melihat berapa rata rata selisih harga akurat dengan harga prediksi, nominal selisih dari MAE pada evaluasi matrik ini dalam bentuk US Dollar**
#
**FORMULA MAE :**
#
![gambar MAE](https://miro.medium.com/max/723/1*9BhnZiaHkApC-gQt3rYpMQ.png)
#
**FORMULA MAPE :**
#
Evaluasi matrik kedua yang akan digunakan adalah Mean Absolute Percentage Error (MAPE). MAPE adalah melakukan penjumlahan secara keseluruhan dengan terlebih dahulu melakukan pengurangan nilai data aktual dengan data prediksi kemudian membaginya dengan data aktual (diharuskan nilainya absolut) dan dikalikan dengan 100 kemudian dibagi dengan banyaknya data yang ada
#
![gambar MAPE](https://media.geeksforgeeks.org/wp-content/uploads/20211120204908/mapeformula.png)
#
Penggunaan MAPE ini untuk melihat berapa persentase rata-rata error antar nilai aktual dengan nilai prediksi
#
**- Karena MAPE menggunakan persentase, berikut gambaran mengenai kategori berdasarkan persentase MAPE :**
#
![gambar MAE](https://data03.123doks.com/thumbv2/123dok/001/615/1615759/7.595.186.438.588.676/tabel-interpretasi-nilai-mape.webp)

# **2. Data Understanding**
#
**Information Kolom**

| **Nama Kolom** | **Deskripsi Kolom** |
| --- | --- |
| longitude | Ukuran seberapa jauh ke barat sebuah rumah; nilai yang lebih tinggi lebih jauh ke barat |
| latitude | Ukuran seberapa jauh ke utara sebuah rumah; nilai yang lebih tinggi lebih jauh ke utara |
| housing_median_age | Usia rata-rata sebuah rumah dalam satu blok (angka yang lebih rendah adalah bangunan yang lebih baru |
| total_rooms | Jumlah total ruangan dalam rumah didalam satu blok |
| total_bedrooms |Jumlah total kamar tidur dalam satu blok |
| population | Jumlah total orang yang tinggal dalam satu blok |
| households | Jumlah total rumah tangga / sekelompok orang yang tinggal dalam satu unit rumah di satu blok  |
| median_income | Pendapatan rata-rata untuk rumah tangga dalam satu blok rumah (diukur dalam puluhan ribu Dolar AS) |
| ocean_proximity | Lokasi rumah dengan laut |
| median_house_value | Nilai median rumah untuk rumah tangga dalam satu blok (diukur dalam Dolar AS) |

# **3. Data Preprocessing**
#
#
- Quick EDA
- Mengecek informasi kolom
- Mengecek jumlah baris dan kolom
- Mengecek data duplikat
- Mengecek missing values
- Mengecek Korelasi data
- Mengecek distribusi data
- Mengecek outlier
---


## 4. **KESIMPULAN**
1. Berdasarkan 5 based model yang sudah di testing prediksi harga rumah, terdapat model algoritma XGBoost memiliki MAE dan MAPE dengan skor errornya terendah sehingga model algoritma ini dijadikan model akhir dan di tuning untuk memaksimalkan performa. Dari hasil model algoritma XGBoost yang dituning, terdapat juga hasil coefisien yang memiliki pengaruh tinggi pada penentuan harga rumah yaitu ada pada feature median_income dan feature ocean_proximity.
<br><br>
2. Model machine learning ini memiliki MAE sekitar 28293.0 US Dollar dengan MAPE 0.168612 (16 %), artinya rata rata harga yang meleset berkisar 28293.0 atau sekitar 16 % dari harga aktual. Pada penerapannya jika harga prediksi lebih tinggi dari pada harga aktual atau harga prediksi lebih rendah dari harga aktual dipengaruhi oleh nilai dari feature median_income dan feature ocean_proximity.
<br><br>
3. Machine learning ini memiliki batasan (**project limitation**) yaitu feature housing_median_age (umur rumah) yang dapat digunakan maksimal 50 tahun dan harga rumah maximal 462200 US Dollar.
<br><br>
4. Dampak dari penggunaan Machine Learning :
   1. Agent property dapat memprediksi harga lebih tepat berdasarkan feature yang tersedia
   2. Jika harga prediksi lebih dari pada harga, maka agent properti dapat menaikkan harga berdasarkan harga prediksi sehingga fee yang diterima juga lebih besar
   3. Penentuan harga lebih stabil karena memiliki dasar penentuan harga yang spesifik
   
## 5. **SARAN**
Hal-hal yang perlu dilakukan untuk mengembangkan model machine learning ini agar lebih akurat :

1. Penambahan feature yang memiliki korelasi langsung dengan harga rumah, misalnya informasi luas tanah, luas bangunan, informasi kondisi bangunan, fasilitas diperkarangan rumah (taman di teras rumah, garasi mobil, tempat menjemur pakaian),fasilitas umum di lingkungan rumah, jarak rumah ke tempat umum (ke tempat transportasi umum, tempat pembelanjaan, tempat ibadah dan rumah sakit), Ukuran kapasitas Watt Listrik dan hal hal yang berkaitan erat dengan fasilitas rumah.
<br><br>
2. Metode hyperparameter dapat ditingkatkan dengan menggunakan metode gridsearch agar dapat mencoba seluruh kombinasi hyperparameter. Karena jika menggunakan randomized search, tidak semua kombinasi hyperparameter dicoba teteapi kita memilih secara acak dari seluruh kemungkinan kombinasi.
<br><br>
1. Untuk mendapatkan hasil yang maksimal pada machine learning (**Project Limitation**) tidak disarankan menggunakan data perumahan yang berumur di atas 50 tahun dan harga rumah tidak diatas 462200 US Dollar karena akan berisiko memiliki error yang sangat tinggi
