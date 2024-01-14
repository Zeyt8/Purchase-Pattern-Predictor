*Analizați rezultatele pe baza tabelelor și graficelor obținute. Ce indică aceste grafice despre puterea de predicție a fiecărui atribut în parte? Cum vă puteți folosi de informația aflată în cadrul unui task de predicție?*

Cu cat corelatia este mai mare, acest atribut influenteaza mai mult eticheta atribuita intrarii. Daca corelatia este foarte mica, am putea elimina acel atribut din setul de date, deoarece influenteaza eticheta foarte putin.

*Comentați asupra rezultatelor, explicând de ce credeți că setup-ul cu rezultat
mai bun pe care îl observați în tabel obține această performanță.*

Arborele de decizie implementat de mine cu max_depth=3 are metricile cele mai bune.

Este de asteptat ca arborele de decizie sa aiba rezultate mai bune decat regresia logistica, deoarece regresia logistica este un model liniar si face aceasta presupunere asupra datelor. Setul nostru de date nu este unul liniar.

Max depth nu influenteaza foarte multe rezultatele, totusi max_depth=3 are rezultate putin mai bune. Cu un max_depth mai mic se evita overfitting-ul.

Cea mai buna varianta de scalare a fost cea fara scalare. Arborii de decizie sunt invarianti la scalarea datelor de intrare, asa ca scalarea este optionala. In varianta fara scalare, modelul este mai putin influentat de outlieri.