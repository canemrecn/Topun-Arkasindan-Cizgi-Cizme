import cv2
import imutils
from collections import deque
import numpy as np
#Bu satırlar, kodun çalışması için gerekli olan kütüphaneleri içe aktarır: cv2 bilgisayar
# görüşü işlemleri için, imutils görüntü işleme yardımcı işlevleri için, deque veri yapısı
# için ve numpy sayısal hesaplamalar için.
dosyaad = ''
#dosyaad değişkeni, işlenecek video dosyasının adını tutar. Şu anda boş bir dize olarak
# ayarlanmıştır.
GENISLIK = 800
NOKTA_SAYISI = 100
#GENISLIK ve NOKTA_SAYISI değişkenleri, görüntünün yeniden boyutlandırılacağı genişlik ve deque
# veri yapısındaki maksimum nokta sayısını belirler.
YESIL=((29, 86, 6), (64, 255, 255))
KIRMIZI=((139, 0, 0), (255, 160, 122))
MAVI=((110, 50, 50), (130, 255, 255))
TURUNCU=((160, 100, 47), (179, 255, 255))
SARI=((10, 100, 100), (40, 255, 255))
#Renk aralıklarını belirleyen değişkenler tanımlanır.
# Örneğin, YESIL değişkeni, yeşil renk aralığını belirler.
altRenk, ustRenk = MAVI
#altRenk ve ustRenk değişkenleri, tespit edilecek nesnenin rengini belirler. Şu anda MAVI
# renk aralığı kullanılmaktadır.
if len(dosyaad) == 0:
    kamera = cv2.VideoCapture(0)
else:
    kamera = cv2.VideoCapture(dosyaad)
#Eğer dosyaad değişkeni boş ise, kamera cihazından video almak için cv2.VideoCapture(0) kullanılır.
# Aksi takdirde, belirtilen dosya adından videoyu okumak için cv2.VideoCapture(dosyaad) kullanılır.
noktalar = deque(maxlen=NOKTA_SAYISI)
#noktalar adlı deque veri yapısı oluşturulur. Bu, algılanan nesnenin merkez noktalarını
# saklamak için kullanılır.
cv2.namedWindow('kare')
cv2.moveWindow('kare', 10, 10)
#'kare' adında bir pencere oluşturulur ve başlangıç konumu ayarlanır.
while True:
    (ok, kare) = kamera.read()
    if len(dosyaad) > 0 and not ok:
        break
#Bu döngü, kameradan veya videodan kareleri okuyarak sürekli çalışır. Eğer dosyaad değişkeni boş
# değilse ve okuma işlemi başarısız olursa döngüyü sonlandırır.
    kare = imutils.resize(kare, GENISLIK)
    hsv = cv2.GaussianBlur(kare, (25,25), 0)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    maske = cv2.inRange(hsv, altRenk, ustRenk)
    maske = cv2.erode(maske, None, iterations=1)
    maske = cv2.dilate(maske, None, iterations=1)
    kopya = maske.copy()
#Görüntüyü yeniden boyutlandırır, bulanıklık uygular ve renk uzayını HSV'ye dönüştürür. Ardından,
# belirtilen renk aralığında bir maske oluşturur, erozyon ve genişleme işlemleri uygular ve maskeyi
# bir kopyaya atar.
    konturlar = cv2.findContours(kopya, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    merkez = None
    if len(konturlar) > 0:
        cmax = max(konturlar, key=cv2.contourArea)
        (x, y), yaricap = cv2.minEnclosingCircle(cmax)
        mts = cv2.moments(cmax)
        merkez = (int(mts['m10']/mts['m00']), int(mts['m01']/mts['m00']))
        if yaricap >= 30:
            cv2.circle(kare, (int(x), int(y)), int(yaricap), (255, 255, 0), 4)
        noktalar.appendleft(merkez)
        for i in range(1, len(noktalar)):
            if noktalar[i] and noktalar[i-1]:
                cizgi_kal = 2
                cv2.line(kare, noktalar[i-1], noktalar[i], (0, 255, 255), cizgi_kal)
    cv2.imshow('kare', kare)
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:
        break
#Bu bölümde, maske üzerindeki konturları bulur ve en büyük konturu seçer. Konturun merkezini ve
# yarıçapını hesaplar. Eğer yarıçap 30'dan büyük ise, bulunan nesnenin etrafına bir daire çizer.
# Merkezi deque veri yapısına ekler. Ardından, deque'deki noktalar arasında çizgi çizer.
#Son olarak, çizilen kareyi ekranda gösterir. Kullanıcının 'q' veya 'ESC' tuşuna basması durumunda
# döngüden çıkar, kamera kaynağını serbest bırakır ve açık olan pencereleri kapatır.
kamera.release()
cv2.destroyAllWindows()
#Bu kod, kullanılan kamera kaynağını serbest bırakır ve açık olan tüm OpenCV pencerelerini kapatır.
# Kamera kaynağı artık kullanılmayacağından serbest bırakılır ve bellek kaynakları serbest bırakılır.
# Aynı şekilde, açık olan pencereler de kapatılarak program sonlandırılır.

