# Próbki tekstu - porównanie modeli

Wszystkie próbki rozpoczęte tym samym ciągiem `"Litwo, ojczyzno moja"`. Pokazujemy najlepszą konfigurację każdego modelu (z najmniejszym val loss).

Porządek: od najprostszego do najbardziej zaawansowanego.


## Litwo, ojczyzno moja… (T=0.7)

### Markov n=1 (bigram) — val=2.568
```
Litwo, ojczyzno mojan gimię m dra! pier wodał.
 wiurwą, wić sa,

Be, kiabówi.
A — przy  ja, waro» gie, ze — Pa rze kżeśw szianaje».
Lewy! patoto
Gałyte wz natw niać durchłoje dnia,
Plut,
Pi zierza makrzę m. dakichybyj — ść.
Led kł!«Kość siczienanią ódania zanielię siem,
Sciżę dkąć:
Aszlęd waróga piem rejątaja
Tobuki Juziów aczywy e ty: bzyśwy skąprzby wy wicza chuż-kiłbódł ł!
Je? odwar zimnie skiądledocedłozlek
Donie
```

### Markov n=3 (sweet spot) — val=2.075
```
Litwo, ojczyzno mojażeby wiodem, ufaRyków, i Zosiu, twą na na biegli mogłem latał mamy;
Przystkie panniknił uczterca ledwielał?
Której służył
Myśliwszyny
Resz kilkuhaj biego dzierzypominał jak mężczykłość jrzadomu;
Czy smuchoci pstwo,
Że w ogrów zawżdymusz pańszczęści grodzierałowo siód; alko Nie chomym dzie!
Bóg rury i w mocne, objąwszególni.
Trzystarażą w któw!
Zbyt suknie, ska krzywachu? tery gdy się jak piętajnie
```

### Markov n=5 (overfit, z back-off) — val=2.696
```
Litwo, ojczyzno moja, po tyby sZastukać;
Sędzia — przeczność liców, przyśpieszy.
Bo wszędzie książki różne stuknie, ja w Peterburku, *borowi tylko grZatem jeden czarny tźli jak i Suwarów jego same obiedny, w talent w las by je własnego klć o jZastukanie chcę wypędził,
Drugi, równi zwyczaj honorowym, sam nieraz się z nią ledwie wspomina i rzeczy)
Wiadomo zakładać głów strumieniło wróżbą ratować!!
Wszystko zrobiłem, ws
```

### Markov n=8 (memorizuje fragmenty) — val=3.279
```
Litwo, ojczyzno moja! ,
Tamten jak źwVulgo W)
Ale komendy jnie,
O Francuz stoi nad rzekę,
Kilku495-9
van der Helle,
Apelujźny ręką, jak gdyby ostatnEuropy

    Francuzi wymóOd złotego runa
On lubił porównywać, a my do kołtuna.
Jeśli prócz tych słyszy, że pan groszaquibusdamskiego, do oRobak obejrza97verend«Na zdrowie jego prośbDobrodzieju — chłox, paów węgrzyna, uprzejmie:
To tak we I tyżeś? do taśmie ze srebra i zło
```

### Linear (best: 02_linear_ctx8_emb32) — val=2.382
```
Litwo, ojczyzno mojada dusteni zbowy i sku z zwadzał Hrabia; Rzokom wióż to u szy dać wajuby z pod wolemi szy ko na mukuszyć przym ogrodki rugiem, żem sięcy skrzu,
Jak wie na z ko fraweraku harzekręda h ruk niemi pole.
Skarzy — le ka ską tak rotow czystę do pierzał powedzie, nodkiej podzieniegoną,
I nie wopiestomił; da
```

### MLP (best: 03_mlp_ctx8_emb16_hid512) — val=2.024
```
Litwo, ojczyzno moja
Rwająstwo i drugie chcaci;
Wojski z waróz szlifę w słuście się oczy rzecie, ale zajezność Prosowaga!
Czy to naskiej szlachta skłankami i krzyką,
Biegnie pozwano, z spóźnił nie złoła,
Żu to z tymko com, pod stoło kamienia dostawy,
Której jeszcze szlachnim nadrzenie pogo, kłót rozpadła z Woźnyu zamie
```

### Transformer 1 head (best: 04_singlehead_b32_e64_h32) — val=2.475
```
Litwo, ojczyzno moja ksi ratwał,
Py sziśłucz turza rbejcze rzem wysiego watym.
Przy kowy więładzcie;
I Ase owi ian odze nierzał si trawaczie
Zaszybądze rzaki wi dziejcie zegięlierzoblete,
Momie mniewak trzy, pram padowaniej te Kustraczeste ciędzertyskiele, szy de sał iczem, ziem dzdak dza nabojesić tały wie j rawać ichał:
Ażaniał poci wać wistrzkanąc aroś wielka potryce nie, wsił miękiemła trzął dziego czo wsterzanie
```

### Mini-GPT (best: 05_minigpt_b128_e192_h6_l6) — val=1.729
```
Litwo, ojczyzno moja wszystkie się zbija.
Aż na Tadeusza, chybiał kwestarz rozmowy,
I w których ciebie ognisków suwały ramiony.
Każdem zdaniał na nowa odparły wojnie,
Uszczuł głowę w pasach szalach na wieczorach;
A gdy na każdy do nim na spuszcza zachodzić,
Uwierzając się w oczach i przecież i kłótnię
Z wesoły z wielkimi do szerokiem osiędza.
Przecież było w górę trąbiono wyszerzędnik,
Wszyscy do księdza szeroki jak na pole,
A widząc do siebie, będących na drugim.

    Cesarz strony w powstał, i trzeba widząc po ra
```

### Mini-GPT EXTENDED 1.2M (10k iter) — val=1.715
```
Litwo, ojczyzno moja.
Wszak się w przypomniej strony w listej swej stary,
Urodził ważne w śród postawie i swe sobie
Te zbiera przymawia koty, o wydawał po okoja.

    Ale w grzesze rycerów kwiaty w kilka zapole,
Szyjąc przy okulesza na głową przeznudze,
Wszystko w środku i wczora w tym był mu boju,
Zosia szczernie z nich do Podkomorzy i wszystko się przed śmiele!
Do ziemnem pana Mapiastu podbiegła pod siedzi,
Choć za swoich i tak swawe służył się noc szyję,
Czy mówiąc z bez mnie młodzież u gwartym sercem,
Jak się w lepszym malarzemu i biegło od sztukał,
Że gdy się on, gdy był to się choć wypol
```

### Mini-GPT EXTENDED 2.7M (10k iter) — val=1.741
```
Litwo, ojczyzno moja siedzą?
Możem jej zapomniać wyprawy ludzki,
Jeśli kozyka, w galop jedynek mówił,
My albo czy wielki z tych lekkim wracać się do młody.
Nie tak przysięgami pomnieć w pomród zwrócił,
Tyle się z Podkomorzy, po co tym rozdziela
Dwa starych jest mam i z miejscem twojej nie mała,
Z wóz sztuką brama na koniec stał podskarnie,
Który często za dni panem na ściany wiecie;
I zaczął wielkich zapala na wyraz się zwrócił,
Na jak nieszczęściem, i masło nas przeszli na dziecinne,
I z kim podniesionymi strażą krzycząc, pod który
Pono słychać było także szczęście przyjacielem.
Tylko słysząc
```


## Litwo, ojczyzno moja… (T=1.0)

### Markov n=1 (bigram) — val=2.568
```
Litwo, ojczyzno mojan gimię m dra! pier wodał.
 wiurwą, wić sa,

Be, kiabówi.
A — przy  ja, waro» gie, ze — Pa rze kżeśw szianaje».
Lewy! patoto
Gałyte wz natw niać durchłoje dnia,
Plut,
Pi zierza makrzę m. dakichybyj — ść.
Led kł!«Kość siczienanią ódania zanielię siem,
Sciżę dkąć:
Aszlęd waróga piem rejątaja
Tobuki Juziów aczywy e ty: bzyśwy skąprzby wy wicza chuż-kiłbódł ł!
Je? odwar zimnie skiądledocedłozlek
Donie
```

### Markov n=3 (sweet spot) — val=2.075
```
Litwo, ojczyzno mojażeby wiodem, ufaRyków, i Zosiu, twą na na biegli mogłem latał mamy;
Przystkie panniknił uczterca ledwielał?
Której służył
Myśliwszyny
Resz kilkuhaj biego dzierzypominał jak mężczykłość jrzadomu;
Czy smuchoci pstwo,
Że w ogrów zawżdymusz pańszczęści grodzierałowo siód; alko Nie chomym dzie!
Bóg rury i w mocne, objąwszególni.
Trzystarażą w któw!
Zbyt suknie, ska krzywachu? tery gdy się jak piętajnie
```

### Markov n=5 (overfit, z back-off) — val=2.696
```
Litwo, ojczyzno moja, po tyby sZastukać;
Sędzia — przeczność liców, przyśpieszy.
Bo wszędzie książki różne stuknie, ja w Peterburku, *borowi tylko grZatem jeden czarny tźli jak i Suwarów jego same obiedny, w talent w las by je własnego klć o jZastukanie chcę wypędził,
Drugi, równi zwyczaj honorowym, sam nieraz się z nią ledwie wspomina i rzeczy)
Wiadomo zakładać głów strumieniło wróżbą ratować!!
Wszystko zrobiłem, ws
```

### Markov n=8 (memorizuje fragmenty) — val=3.279
```
Litwo, ojczyzno moja! ,
Tamten jak źwVulgo W)
Ale komendy jnie,
O Francuz stoi nad rzekę,
Kilku495-9
van der Helle,
Apelujźny ręką, jak gdyby ostatnEuropy

    Francuzi wymóOd złotego runa
On lubił porównywać, a my do kołtuna.
Jeśli prócz tych słyszy, że pan groszaquibusdamskiego, do oRobak obejrza97verend«Na zdrowie jego prośbDobrodzieju — chłox, paów węgrzyna, uprzejmie:
To tak we I tyżeś? do taśmie ze srebra i zło
```

### Linear (best: 02_linear_ctx8_emb32) — val=2.382
```
Litwo, ojczyzno mojakł:
Ma, czubażą zborocość, webszić ną kom ała
  n no pwienie zy,ma drącznie premianni;
Tadaje Mógłam  przedli mruszwie postu: palem, a ile naj kie ci o m pie j wy śni dzo wadzoć!
Nieszakdzi k Ojojąc wim czyktam
Dubrzywk!»
Prewi, czkie  ogodomyłuły, i łógnę wiele monę towie nko drazi jusik o wroliwad
```

### MLP (best: 03_mlp_ctx8_emb16_hid512) — val=2.024
```
Litwo, ojczyzno mojanie fanaki…
Dosłanie mój (klutka pałaśnie jak ogajedni*);
Dleżą się gmały
Choć swet gam panier z ptórejgo kroteś wielka wielkie sza, z jest zrobić pozwania na Horeszkój?… z ku tniki barznie potrupaci, dał wimskiej nie mu Barsknął,
I po wpadł, że Wielki ibruwa
W kradku swe dziwiąc sywia szuch ga zbro
```

### Transformer 1 head (best: 04_singlehead_b32_e64_h32) — val=2.475
```
Litwo, ojczyzno moja zyweszbierspo powi oć wy odysła.

Wzłyżoma; ziespru,
Wyłosto powię
Wabaja, rok wzi, kodwie zwy wodałą gł pość zkuł lie iak, proście prówny dwysiubrzmięłko si młemi
Motała ni nenienieska w knod przył sku, mieczi, ga niu du tanopierwe, dustrodelkę)
I cię w czdyrazy zatrwa ni, st trodzę, arzy spajdyk wse szebradzekzwy; wszem ją waniączy kaczyskał śrybruć,
A daw seli, mni?
I pry legieświędomłotu.
Chc
```

### Mini-GPT (best: 05_minigpt_b128_e192_h6_l6) — val=1.729
```
Litwo, ojczyzno moja wieczaje;
Z nim jeśli możnym powiecie: niech pośpieszył?»

    Już wygrom z cośmi, za wracać z dala sukniu!
Nikt będąc między z nim domu w koniach domowyra!
Bardzo policąc rękami, gdy jak wiasności
Wesoło żniazie chciałoś się w pole urzędy,
Że myśląc ci kędy żelazom je w tej roztwiedzieć.

    A wiedziała szczlachta jezorał z lekka;
Pano o judro galinę była progi:
Królewaj się opowrokiem, a to rączka szedłum,
Gdy rzekł, patrzą; żelizu Brzytewka honono».

    O Sędzia Panna: «Pan Tadeusz ty! Ger
```

### Mini-GPT EXTENDED 1.2M (10k iter) — val=1.715
```
Litwo, ojczyzno moja i jada czeka:
Chcąc Gerwazy wojsków! aż z kądzieli tak konopole
Czyż tajemno do krewne szturmem słany,
Podwolała się najwinne wielkim widni, że mu wyraźnie!
Jakoś u daleki mocnie czas w tych szlachciców,
Jak wilk w prawów się cudziemu chartów kolę,
Które bija w okapewności, a pod nich świeżodniejszy,
A moi cała dzieckomo, ptaki stadał,
Król na każdej się, Sędzia nad szlachcic ze szlachciców honorowie,
A do której mercian nie pęk człowiek ludzi służą,
Którą będą się od skarczmy; odlecz się, a zura,
Drzewa się pod gromadę dzisiąć błystki,
Odmyknął więc, lecz tak u boru. Kluc
```

### Mini-GPT EXTENDED 2.7M (10k iter) — val=1.741
```
Litwo, ojczyzno moja drugi.
Je nie był go wróble naszą obszczęty,
Krwią i jak z cuga dzwon Sakiem dostały,
Odparłszy do skutku, takąwszy wzgląd do nich Wojski.
W Aoniec już nas zjawił się chwał się nad rozmową,
I jeszcze zawsze się z przodu jakby w domu.

    Do zawiesz przy nim swoich domu zgromadzenie.
Ostróżny trzykroć od ust to wielkiego nogami,
A w polu ukazał skargy bujać i stuki.

    W tatykich stary płamiętą deszkole dlintą.
Tam dawni procesów nim nie corgi przy zważa;
Bo znam cnudzić; najpiękniejszej anikowie!»

    «Ekonom został Maciek, a tylko nie działo,
Nie dziwa iskopoląc zrobi
```
