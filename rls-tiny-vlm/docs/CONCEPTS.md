# Ce que tu dois savoir expliquer (entretien) — avec renvoi au code

## 1. Le flux complet, avec les formes
`image (B,3,64,64) uint8` → encodeur CNN → `(B,128,8,8)` → adapter (flatten+Linear) → `(B,64,128)` visual tokens
→ concat avec embeddings des lettres `(B,L,128)` → décodeur → `(B,64+L,128)` → tête Linear → on garde les positions
à partir du dernier token visuel → `logits (B, L+1, 27)`. (`src/model/vlm.py`)

**Pourquoi L+1 sorties ?** Il n'y a pas de token de début : la sortie du dernier token visuel (position 63) prédit la
1re lettre. La sortie après la dernière lettre prédit `<eos>`. Donc `targets = [w1..wn, <eos>]` (`src/tokenizer.py`).
Le décalage "input = lettres, target = lettre suivante" est implicite dans cette position.

## 2. Tailles de CNN
`out = floor((n + 2p − k)/s) + 1`. Conv 3×3, s=1, p=1 → taille inchangée. MaxPool 2×2 → /2. 64→32→16→8 : grille 8×8 = 64 tokens.
Une feature map contient, à chaque position (x,y), un vecteur de 128 canaux décrivant le contenu local (couleur, bord, forme).
Le pooling perd la position exacte dans chaque cellule 2×2 ; la position globale revient via l'embedding positionnel du décodeur.

## 3. Attention écrite à la main (`src/model/attention.py`)
- Q, K, V = trois projections linéaires de `x` (ici une seule `Linear(D,3D)` puis `chunk`). Q "ce que je cherche", K "ce que j'offre", V "ce que je transmets".
- Séparer en têtes : `(B,T,D) → (B,H,T,dh)`, chaque tête travaille sur un sous-espace de dimension `dh = D/H`.
- Scores `QKᵀ/√dh` : `(B,H,T,T)`. Sans `√dh`, la variance des scores vaut `dh`, le softmax sature (quasi one-hot) et les gradients s'effondrent.
- Masque : `masked_fill(~mask, -inf)` AVANT softmax → poids 0 exactement. Une ligne entièrement masquée donnerait NaN (jamais le cas ici : la diagonale est toujours visible).
- `softmax(dim=-1)` puis `@ V`, fusion des têtes, projection de sortie.
- Mémoire ∝ `B·H·T²` : la matrice T×T est matérialisée et gardée pour le backward → c'est ce que les kernels fusés (FlashAttention) évitent.

## 4. Le masque "prefix-LM" (`src/model/decoder.py`)
Tokens visuels : se voient tous entre eux (bidirectionnel), ne voient jamais les lettres. Lettre *i* : voit toute l'image et les lettres ≤ *i* (causal).
Pourquoi : l'image n'a pas d'ordre temporel ; interdire le futur sert seulement aux lettres, sinon le modèle recopie la réponse (le test
`test_letter_logits_do_not_see_future_letters` le vérifie). Option E3 : `visual_causal: true`.

## 5. Bloc décodeur
Pre-LN : `x = x + Attn(LN(x))`, `x = x + MLP(LN(x))`. Les résidus laissent passer le gradient (chemin identité) ; LayerNorm stabilise l'échelle ;
le MLP (D→4D→D) transforme chaque position indépendamment ; l'attention est le seul endroit où les positions échangent de l'information.
Embeddings de position appris (la permutation de tokens est invisible à l'attention sans eux).

## 6. Autograd / backprop (`src/train.py`)
- Forward : construit le graphe de calcul, chaque opération garde ce dont son backward a besoin (activations).
- `optimizer.zero_grad()` : remet `p.grad` à zéro/None (sinon les gradients S'ACCUMULENT d'un pas à l'autre).
- `loss.backward()` : règle de la chaîne du résultat vers les feuilles ; remplit `p.grad` de chaque paramètre.
- `optimizer.step()` : AdamW met à jour `p` à partir de `p.grad` (moments m, v : 2 buffers par paramètre → mémoire).
- Loss : `cross_entropy(..., ignore_index=-100)` : le padding n'entre ni dans la moyenne ni dans le gradient.

## 7. Lire un entraînement
E0 (un batch) → loss ≈ 0 sinon BUG. Train baisse + val monte = sur-apprentissage. Les deux stagnent haut = bug ou modèle trop petit / lr mauvais.
Loss initiale attendue ≈ ln 27 ≈ 3,3 (le premier pas de notre run affiche 3,33 : bon signe).

## 8. Précision lettre vs vrai résultat (le "piège" de la consigne)
Dans `circle`, après `c` les lettres `ircle` sont devinables par l'orthographe seule. Un modèle aveugle obtient donc une grande précision
lettre mais ~0 d'exact match. `evaluate.py` sépare : première lettre d'une partie (là où l'image compte) vs lettres à l'intérieur (orthographe).
C'est précisément le résultat de E1 : l'écart entre précision lettre et précision d'attribut.

## 9. GPU et mesure (S1, niveau 2)
- Asynchrone : les appels CUDA reviennent avant la fin du calcul → `torch.cuda.synchronize()` avant de lire le chronomètre (`benchmarks/common.py`).
- Warm-up : contexte CUDA, autotune cuDNN, allocateur, chargement paresseux des kernels → on jette les premières itérations.
- Petit batch : le GPU est limité par l'overhead (lancement de kernels, Python) et non par le calcul : le CPU peut gagner ; au-delà d'un certain batch le GPU domine.
  Le point de croisement est ce que tu dois mesurer et expliquer (compute / bande passante mémoire / overhead — Horace He).
- Mémoire pic = paramètres + gradients + état Adam (2×) + activations (seule partie qui croît avec le batch). `memory.py` : estimation PUIS mesure.
- `bench` exclut le chargement des données du chrono principal (déclaré) ; la variante `with_h2d` inclut la copie CPU→GPU.

## 10. Questions d'entretien probables
1. Pourquoi 27 tokens ? Pourquoi pas de token de début ni de padding ? 2. Que se passe-t-il si on oublie le masque ? √dk ? 3. Quelle forme a chaque tenseur ?
4. Pourquoi le test d'équivalence avec SDPA ? 5. Pourquoi le modèle aveugle a 60 %+ de lettres justes ? 6. Pourquoi la précision baisse sur `test_heldout` ?
7. Pourquoi le GPU est plus lent à batch 1 ? 8. Modifie en live : masque causal sur les tokens visuels, 2 poolings au lieu de 3 (→16×16 = 256 tokens), nombre de têtes.
