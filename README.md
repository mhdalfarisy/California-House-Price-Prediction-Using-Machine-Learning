## PROJECT : California House Price Prediction Using Machine Learning

![a1](https://github.com/mhdalfarisy/California-House-Price-Prediction-Using-Machine-Learning/blob/main/gambar/CA-Sales-Home-Volume.png)

- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Analytical Approach](#analytical-approach)
- [Metric Evaluation](#metric-evaluation)
- [Definisi Kolom](#definisi-kolom)
- [Machine Learning](#machine-learning)
  - [Modeling Machine Learning](#modeling-machine-learning)
  - [Saran](#saran)
  - [Kesimpulan](#kesimpulan)
- [Others Data Visualization Report](#others-data-visualization-report)

### **Problem Statement**

Pengaruh dari kurang tepatnya dalam penentuan harga rumah ini berdampak pada penurunan penjualan agent properti yang menyebabkan harga tidak stabil dan customer tidak mampu membeli karena harganya yang naik secara drastis. Hal menjadi tantangan untuk agent properti untuk **menentukan harga rumah yang dijual secara tepat**, karena dalam menentukan harga rumah memiliki banyak faktor yang mendukung penentuan tinggi atau rendahnya harga rumah seperti lokasi rumah, umur rumah,jumlah ruangan,jumlah kamar tidur, keramaian lingkungan rumah, kawasan perumahan elit atau tidak dan jarak rumah dengan lokasi wisata. Tidak cukup hanya dengan membandingkan kondisi rumah yang akan dijual dengan rumah lainnya yang berada disekitar kemudian langsung menentukan harga rumah dengan harga pasar atau langsung menentukan hanya mengikuti harga pasar.

### **Objectives**
#
Berdasarkan permasalahan diatas, agen properti memerlukan **perangkat yang dapat membantu memperediksi harga rumah secara tepat**, perangkat ini disebut dengan machine learning. Machine learning ini dapat agen properti *gunakan ketika client (pihak pemilik rumah) ingin menjual rumahnya dan agen properti sebagai perantara* dapat menggunakan Machine learning ini untuk memprediksi harga rumah yang sesuai berdasarkan faktor-faktor lokasi rumah, umur rumah,jumlah ruangan,jumlah kamar tidur, keramaian lingkungan rumah, kawasan perumahan elit atau tidak dan jarak rumah dengan lokasi wisata dengan menggunakan machine learning.

### **Analytical Approach**

Jadi, dalam hal ini yang perlu dilakukan : 
1. Menganalisis data untuk dapat menemukan pola dari fitur-fitur yang ada, yang membedakan satu perumahan dengan yang lainnya
2. Membangun model regresi yang dapat membantu agent properti untuk menyediakan sebuah perangkat prediksi harga rumah
3. Menemukan fitur yang memiliki pengaruh besar terhadap target **[median_house_values]** agar model yang dibangun lebih sederhana dan memiliki tingkat akurat yang baik

### **Metric Evaluation**
#
Evaluasi matrik pertama yang akan digunakan adalah Mean Absolute Error (MAE). Mean Absolute Error adalah rataan nilai absolut dari error yang dihasilkan oleh model regresi. Jika nilai yang di hasilkan dari evaluasi metrik MAE 0 (nol) artinya akurasi dari hasil prediksi harga rumah mendekati akurat dan jika nilai dari MAE 1 (satu) maka akurasi dari prediksi dari hasil prediksi harga rumah tidak mendekati nilai yang akurat.

Penggunaan evaluasi matrik **MAE untuk melihat berapa rata rata selisih harga akurat dengan harga prediksi, nominal selisih dari MAE pada evaluasi matrik ini dalam bentuk US Dollar**

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


### **Definisi Kolom**
#

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

### **Machine Learning**

#### **Modeling Machine Learning**

![m1](https://github.com/mhdalfarisy/California-House-Price-Prediction-Using-Machine-Learning/blob/main/gambar/training%20before%20tuning.JPG)

Ketika dilakukan training pada model machine learning, **Training Model XGBoost Regressor** memiliki nilai error dari *MEAN* dan *MAPE* terendah, dan karena sudah mendapatkan gambaran tentang training datanya, kemudian dilakukan testing data untuk mendapatkan base model nya.

![m2](https://github.com/mhdalfarisy/California-House-Price-Prediction-Using-Machine-Learning/blob/main/gambar/testing%20before%20tuning.JPG)

Ketika dilakukan testing terdapat **Testing Model XGBoost Regressor** nilai error terendah *MEAN* dan *MAPE* sehingga model ini menjadi base model machine learning. Selanjutnya akan dilakukan tunning untuk menurunkan angka error nya. **Hyperparameter tuning XGBoost Regressor** yang digunakan adalah :

- **max_depth** = merupakan kedalaman maksimum pohon, ini digunakan untuk mengontrol over-fitting karena kedalaman yang lebih tinggi akan memungkinkan model untuk mempelajari hubungan yang sangat spesifik untuk sampel tertentu.
- **learning_rate** = Hal ini analog dengan kecepatan belajar, penyusutan ukuran langkah yang digunakan dalam pembaruan untuk mencegah overfitting.
- **n_estimators** = Jumlah pohon yang ingin bangun sebelum mengambil voting maksimum atau rata-rata prediksi. Jumlah pohon yang lebih tinggi memberi kinerja yang lebih baik tetapi membuat running laptop sedikit lama.
- **subsample** = Jumlah baris tiap pohon
- **gamma** = Node dipecah hanya ketika hasil split memberikan pengurangan nilai positif,semakin besar nilainya, semakin konservatif/simpel modelnya
- **colsample_bytree** = Jumlah feature subsampel kolom saat membangun setiap pohon. Subsampling terjadi sekali untuk setiap pohon yang dibangun
- **reg_alpha** = Digunakan dalam kasus dimensi yang sangat tinggi sehingga algoritma berjalan lebih cepat saat diimplementasikan.
Peningkatan nilai ini akan membuat model lebih konservatif.

Dalam melakukan tuning, disini menggunakan **RandomizedSearchCV** karena pada hyperparameter xgb saya banyak menggunakan model parameternya sehingga randomized search digunakan untuk melakukan komputasi lebih efisien.

Hasil sebelum tuning 

![m4](https://github.com/mhdalfarisy/California-House-Price-Prediction-Using-Machine-Learning/blob/main/gambar/before%20tuning.JPG)

Hasil sebelum tuning memiliki MAE dengan total 29072 USD dengan persentase MAPE 0,17%

Hasil setelah tuning

![m3](https://github.com/mhdalfarisy/California-House-Price-Prediction-Using-Machine-Learning/blob/main/gambar/after%20tuning.JPG)

Setelah dilakukan tuning error MAE mengalami penurunan menjadi 28293 USD dengan total persentase error 0,16%.

perbedaan menggunakan machine learning dan tidak menggunakan machine learning :

![m6](https://github.com/mhdalfarisy/California-House-Price-Prediction-Using-Machine-Learning/blob/main/gambar/efek%20mengggunakan%20machine%20learning.JPG)

Dari sample perbandingan data dengan harga sebelm prediksi dan harga setelah prediksi terdapat perbedaan. Harga prediksi lebih tinggi, namun perbedaan ini didukung oleh feature feature yang memiliki pengaruh tinggi terhadap target. Disini agent properti dapat menggunakan nilai dari data aktual untuk menentukan harga rumah dapat juga menggunakan harga prediksi, jika menggunakan harga prediksi akan mendapatkan keuntungan lebih dari total selisih dengan harga aktual selain itu semakin tinggi harga yang ditentukan (menggunakan harga prediksi) berdasarkan spesifikasi perumahannya maka semakin besar juga fee yang diterima agent properti. Dalam hal ini spesifikasinya berkaitan dengan feature yang ada pada dataset.

- Dari sample perbandingan data di atas, terdapat 2 index dengan nilai prediksi yang naik dan turun. Penjelasan sebagai berikut :
    - Index 1 **(Case Harga prediksi lebih tinggi)** : Median_house_values bernilai 100000.0 US Dollar, Harga_Prediksi_Rumah bernilai 104906.0 US Dollar dan selisih antara harga aktual dengan harga prediksi 4906.0 US Dollar. Artinya lokasi perumahan yang dijual adalah perumahan di kawasan elit yang penduduknya memiliki median_income yang tinggi dan berlokasi jauh dari laut sehingga harga dari prediksi mengalami kenaikan.
    - Index 2 **(Case Harga prediksi lebih rendah dari harga aktual)** : Median_house_values bernilai 285800.0 US Dollar, Harga_Prediksi_Rumah bernilai 276521.0 US Dollar dan selisih antara harga aktual dengan harga prediksi 9279.0 US Dollar. Artinya lokasi perumahan yang dijual bukan dikawasan elit yang penduduknya tidak memiliki pendapatan yang tinggi dan berlokasi dekat dengan laut.

#### **Kesimpulan**
1. Perumahan yang paling banyak di huni berasal dari daerah <1H Ocean
2. Harga perumahan tertinggi pada saat itu untuk daerah <1H Ocean berkisar 500000 USD
3. Berdasarkan 5 based model yang sudah di testing prediksi harga rumah, terdapat model algoritma XGBoost memiliki MAE dan MAPE dengan skor errornya terendah sehingga model algoritma ini dijadikan model akhir dan di tuning untuk memaksimalkan performa. Dari hasil model algoritma XGBoost yang dituning, terdapat juga hasil coefisien yang memiliki pengaruh tinggi pada penentuan harga rumah yaitu ada pada feature median_income dan feature ocean_proximity (INLAND).
<br><br>
2. Model machine learning ini memiliki MAE sekitar 28293.0 US Dollar dengan MAPE 0.168612 (16 %), artinya rata rata harga yang meleset berkisar 28293.0 atau sekitar 16 % dari harga aktual. Pada penerapannya jika harga prediksi lebih tinggi dari pada harga aktual atau harga prediksi lebih rendah dari harga aktual dipengaruhi oleh nilai dari feature median_income dan feature ocean_proximity.
<br><br>
3. Machine learning ini memiliki batasan (**project limitation**) yaitu feature housing_median_age (umur rumah) yang dapat digunakan maksimal 50 tahun dan harga rumah maximal 462200 US Dollar.
<br><br>
4. Dampak dari penggunaan Machine Learning :
   1. Agent property dapat memprediksi harga lebih tepat berdasarkan feature yang tersedia
   2. Jika harga prediksi lebih dari pada harga, maka agent properti dapat menaikkan harga berdasarkan harga prediksi sehingga fee yang diterima juga lebih besar
   3. Penentuan harga lebih stabil karena memiliki dasar penentuan harga yang spesifik
   
#### **Saran**
Hal-hal yang perlu dilakukan untuk mengembangkan model machine learning ini agar lebih akurat :

1. Penambahan feature yang memiliki korelasi langsung dengan harga rumah, misalnya informasi luas tanah, luas bangunan, informasi kondisi bangunan, fasilitas diperkarangan rumah (taman di teras rumah, garasi mobil, tempat menjemur pakaian),fasilitas umum di lingkungan rumah, jarak rumah ke tempat umum (ke tempat transportasi umum, tempat pembelanjaan, tempat ibadah dan rumah sakit), Ukuran kapasitas Watt Listrik dan hal hal yang berkaitan erat dengan fasilitas rumah.
<br><br>
2. Metode hyperparameter dapat ditingkatkan dengan menggunakan metode gridsearch agar dapat mencoba seluruh kombinasi hyperparameter. Karena jika menggunakan randomized search, tidak semua kombinasi hyperparameter dicoba teteapi kita memilih secara acak dari seluruh kemungkinan kombinasi.
<br><br>
1. Untuk mendapatkan hasil yang maksimal pada machine learning (**Project Limitation**) tidak disarankan menggunakan data perumahan yang berumur di atas 50 tahun dan harga rumah tidak diatas 462200 US Dollar karena akan berisiko memiliki error yang sangat tinggi

<br>

# **OTHERS PROJECT :**

<br>
<table>
<tbody>
 <tr>

<h1 align="left">Data Analyst</h1>
  
<td align="left" width="50%">
<span><b><Left>E-Commers Pakistan</center></b></span> 
<code><a href="https://github.com/mhdalfarisy/EDA---Pakistan-s-Larges-Ecommers" target="_blank">
<img height=250px src="https://github.com/mhdalfarisy/EDA---Pakistan-s-Larges-Ecommers/blob/main/Images/62253a402fccf.jpg"> 
</td>
<!-- <tr> -->
<td align="left" width="50%">
<span><b><Left>Employee Attrition</center></b></span> 
<code><a href="https://github.com/mhdalfarisy/Employee-Analysis-Attrition-Report" target="_blank">
<img height=250px src="https://github.com/mhdalfarisy/Employee-Analysis-Attrition-Report/blob/main/Aset/Reasons-Attrition1_large%20(1).jpg"> 
</td>
</tbody>
</table>
 <tr>
<br>
<table>
<tbody>
 <tr>
 
<h1 align="left">Machine Learning</h1>

<td align="left" width="20%">
<span><b><left>California House Price</center></b></span> 
<code><a href="https://github.com/mhdalfarisy/California-House-Price-Prediction-Using-Machine-Learning" target="_blank">
<img height=150px src="https://github.com/mhdalfarisy/California-House-Price-Prediction-Using-Machine-Learning/blob/main/gambar/CA-Sales-Home-Volume.png"> 
</td>

<td align="left" width="20%">
<span><b><left>Credit Card Fraud</center></b></span> 
<code><a href="https://github.com/mhdalfarisy/Credit-Card-Fraud-Prediction" target="_blank">
<img height=150px src="https://github.com/mhdalfarisy/Credit-Card-Fraud-Prediction/blob/main/68747470733a2f2f65787465726e616c2d636f6e74656e742e6475636b6475636b676f2e636f6d2f69752f3f753d687474707325334125324625324661692d6a6f75726e65792e636f6d25324677702d636f6e74656e7425324675706c6f61647325324632303139253246.jfif"> 
</td>

<!-- <tr> -->
<td align="left" width="20%">
<span><b><left>Telco Customer Churn</center></b></span> 
<code><a href="https://github.com/mhdalfarisy/Employee-Promotion" target="_blank">
<img height=150px src="https://github.com/mhdalfarisy/mhdalfarisy/blob/main/7-Strategies-To-Reduce-Customer-Churn-Rate.png"> 
</td>

<!-- <tr> -->
<td align="left" width="20%">
<span><b><left>Employee Promotion</center></b></span> 
<code><a href="https://github.com/mhdalfarisy/Telco-Customer-Churn-Predict" target="_blank">
<img height=150px src="https://github.com/mhdalfarisy/mhdalfarisy/blob/main/advance-career.jpg"> 
</td>

</tbody>
</table>
 <tr>
  
<br>

<table>
<tbody>
 <tr>

<h1 align="left">Data Visualization</h1>
  
<td align="left" width="30%">
<span><b><Left>E-Commers Pakistan</center></b></span> 
<code><a href="https://public.tableau.com/app/profile/muhammad.al.farisy6147/viz/ProjectE-CommersPakistanDashboard/Dashboard1" target="_blank">
<img height=200px src="https://github.com/mhdalfarisy/mhdalfarisy/blob/main/Pakistan%20Visualisasi.png"> 
</td>
 
<!-- <tr> -->
<td align="left" width="30%">
<span><b><left>Employee Attrition Report</center></b></span> 
<code><a href="https://public.tableau.com/app/profile/muhammad.al.farisy6147/viz/ProjectHumanResourceAttritionAnalysisDashboard/Dashboard1" target="_blank">
<img height=200px src="https://github.com/mhdalfarisy/mhdalfarisy/blob/main/HRD%20VIsualisasi.png"> 
</td>
 
<!-- <tr> -->
<td align="left" width="25%">
<span><b><left>Telco Customer Churn</center></b></span> 
<code><a href="https://public.tableau.com/app/profile/muhammad.al.farisy6147/viz/CustomerChunVisualization/Dashboard2?publish=yes" target="_blank">
<img height=200px src="https://github.com/mhdalfarisy/mhdalfarisy/blob/main/Telco%20Customer%20%2022.png"> 
</td>
 
</tbody>
</table>
 <tr>



<br>
<!-- <h1 align="center">Others Data Visualization Report</h1> -->
<td align="left" width="30%">
<span><b><left>Others Data Visualization Report (Click Picture) :   </left></b></span> 
<code><a href="https://public.tableau.com/app/profile/muhammad.al.farisy6147" target="_blank"><img height="100" src="https://github.com/mhdalfarisy/mhdalfarisy/blob/main/tol_devices_optimized.png"></a></code>
<br>
<br>
<br>
 
**💬 Ask me about anything, I'll be happy to help!** <br>
**💬 My inbox is always open, Contact me**
<br>
<br> 
  </a>
  <a href="mailto:m.alfarisy797@gmail.com">
    <img align="left" alt="Muhammad Al-farisy | Gmail" width="26px" src="https://cdn.worldvectorlogo.com/logos/official-gmail-icon-2020-.svg" />
  </a>
  <a href="https://www.linkedin.com/in/m-alfarisy97/">
    <img align="left" alt="Muhammad Al-farisy | LinkedIN" width="26px" src="https://cdn.worldvectorlogo.com/logos/linkedin-icon-2.svg" />    
  </a>
<br>
<br>


| <a href="https://github.com/mhdalfarisy"><img align="center" src="https://github-readme-stats.vercel.app/api?username=mhdalfarisy&show_icons=true&include_all_commits=true&theme=buefy&hide_border=true" alt="Anurag's github stats" /></a> | <a href="https://github.com/mhdalfarisy/github-readme-stats"><img align="center" src="https://github-readme-stats.vercel.app/api/top-langs/?username=mhdalfarisy&layout=compact&theme=buefy&hide_border=true" /></a> |
| ------------- | ------------- |
 
<table>
<tbody>
 <tr>
 
<h1 align="left">THANKS YOU !!! </h1>

<td align="center" width="30%">
<img height=300px src="https://media.giphy.com/media/dyzew7Py7bnW9DiJJj/giphy.gif"> 
</td>  
  
<!-- <td align="center" width="30%">
<img height=300px src="https://media.giphy.com/media/7c8QeB0VMddFOuu4iR/giphy.gif"> 
</td>
  
<td align="right" width="30%">
<img height=300px src="https://media.giphy.com/media/gutZ5Pm6Xl62eIf5RZ/giphy.gif"> 
</td>    -->

⭐️ From [Muhammad Al-farisy](https://github.com/mhdalfarisy)
