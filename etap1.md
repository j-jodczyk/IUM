<!-- i)definicję problemu biznesowego, zdefiniowanie zadania/zadań modelowaniai wszystkich założeń, zaproponowania kryteriów sukcesu),ii)analizę danych z perspektywy realizacji tych zadań(trzeba ocenić, czy dostarczone dane są wystarczające –może czegoś brakuje, może coś trzeba poprawić,domagać się innych danych, ...), -->

# Etap 1
### autorzy
- Julia Jodczyk
- Filip Pawłowski

## Zadanie:
W ramach projektu wcielamy się w rolę analityka pracującego dla portalu„Pozytywka” – serwisu muzycznego, który swoim użytkownikom pozwala na odtwarzanie ulubionych utworów online. Praca na tym stanowisku nie jest łatwa –zadanie dostajemy w formie enigmatycznego opisu i to do nas należy doprecyzowanie szczegółów tak, aby dało się je zrealizować. To oczywiście wymaga  zrozumienia  problemu,  przeanalizowania  danych,  czasami  negocjacji  z  szefostwem. Same  modelemusimy skonstruowaćtak,  aby gotowe  były do  wdrożenia  produkcyjnego – pamiętając,  że  w  przyszłości  będą  pojawiać  się  kolejne  ich  wersje,z  którymi  będziemy eksperymentować.

“Jakiś czas temu wprowadziliśmy konta premium, które uwalniają użytkowników od słuchania reklam. Nie są one jednak jeszcze zbyt popularne – czy możemy się dowiedzieć, które osoby są bardziej skłonne do zakupu takiego konta?”

## Definicja problemu biznesowego
Z biznesowego punktu widzenia, chcemy móc identyfikować użytkowników skłonnych do zakupu konta premium, aby bardziej skupić się na tej grupie pod względem reklamowania kont premium. Proces identyfikacji powinien odbywać się w pewnych odstępach czasowych - być może zainteresowanie użytkowników kontem premium w zależności od różnych czynników się zmienia.

## Zadanie modelowania
Zadaniem modelowania będzie klasyfikacja binarna:
1. użytkownicy, którzy kupili konto premium
2. użytkonicy, którzy nie kupili konta premium.


## Założenia
- Dane będą analizowane w odstępach czasowych, które zdefiniujemy po analizie danych. Oznacza to, że jeśli użytkownik kupi konto premium w n-tym analizowanym okresie, to zostaje wykluczony z analizy w okresach następujących po nim.
- Głównym źródłem analizy będą historie sesji użytkowników (jak długo przebywają na platformie, z jaką częstotliwością wyświetlają się im reklamy, itp.).
- Pod uwagę weźmiemy również pozostałe cechy użytkowników - ulubione gatunki i miejsce zamieszkania.

## Kryteria sukcesu
- Lepsza niż ślepy strzał predykcja tego, czy użytkownik w danym okresie czasu kupi konto premium.

## Analiza danych