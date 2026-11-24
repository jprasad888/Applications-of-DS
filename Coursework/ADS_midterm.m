function ADS_midterm()
%ADS_MIDTERM  Mid-term pipeline for CMP-L014-0 Applications of Data Science.
%
% Problem 2: extracting urban infrastructure from satellite images.
% Question : can hand-crafted (non-deep) image features tell ROAD patches from
%            BUILDING patches in multi-source satellite imagery?
%
% Tasks covered here (mid-term scope):
%   Task 2 - refine the segmentation dataset into a balanced 2-class patch set
%   Task 3 - mathematical feature engineering (34 features, all hand-coded), built on
%            the module labs: morphology (Wk 2), median/Gaussian filters (Wk 3),
%            padding and Otsu (Wk 4), Kapur and multi-level thresholds (Wk 5),
%            plus Laplacian edges and a coarse histogram of oriented gradients (HOG)
%   Task 4 - descriptive statistics, plus inferential tests (Welch t, Mann-Whitney U, 95% CI)
%   Task 5 - visualisation of the dataset and the features (part 1)
%   PCA    - dimensionality check, and a PCA-reduced feature vector for the ML stage
%            (Task 6): the model on the first k PCs is compared with the model on all features
%   XAI    - preview of Task 6: a white-box logistic regression validated by image,
%            explained with feature importance and counterfactuals (Week 1 XAI methods)
%            and tested with ROC/AUC, a Wilson CI for accuracy and an exact binomial test
%
% INPUT : DATA_DIR containing images/imageNNNN.jpg and masks/maskNNNN.png
%         (Kaggle "Multi-Source Satellite Imagery for Segmentation", CC0).
%         If DATA_DIR is missing and DATA_URL is set, the zip is downloaded first.
% OUTPUT: RESULTS_DIR with figures (*.png), tables (*.csv) and features.mat
%
% BUILT-IN POLICY
%   No MATLAB maths or toolbox functions are used. Every method is written by hand
%   below: nearest-colour labelling, random sampling, median and Gaussian filtering,
%   Sobel, Laplacian, HOG, Otsu/Kapur/multi-level thresholds, morphology, GLCM, 2-D DFT,
%   moments, statistical tests, kernel density, Jacobi PCA, logistic regression, k-NN,
%   naive Bayes, LDA, linear and RBF-kernel SVMs, CART tree, random forest, McNemar
%   test, permutation importance, counterfactuals, log-gamma and erfc.
%   The basic maths is hand-written too (last section of this file): sums, running
%   sums, max/min, abs, floor, mod, find, square root, exp, log, atan2, sine and
%   cosine, a linear-equation solver, and the small grid helpers. What remains is the
%   language itself (+ - * / .^, comparisons, indexing, loops), the array tools size,
%   numel, zeros, ones, true, false, reshape and permute, file reading and writing,
%   and plotting.

% Runs in MATLAB R2016b or later.
% Author: Aditya Vardhan Sivalanka (A00091352)     Date: November 2026

%% ---------------------------------------------------------------- settings
codeDir     = fileparts(mfilename('fullpath'));
% Project folder. In the repo the code sits in code/ with the dataset in ../all;
% in MATLAB Online (MATLAB Drive) the script sits next to all/ or all.zip.
if exist(fullfile(codeDir, '..', 'all'), 'dir')
    baseDir = fullfile(codeDir, '..');
else
    baseDir = codeDir;
end
DATA_DIR    = fullfile(baseDir, 'all');             % dataset folder: all/images, all/masks
% Public Kaggle download of the same CC0 dataset (version 1, 573 MB, unzips to all/images + all/masks)
DATA_URL    = ['https://www.kaggle.com/api/v1/datasets/download/hammadjavaid/' ...
               'multi-source-satellite-imagery-for-segmentation?datasetVersionNumber=1'];
RESULTS_DIR = fullfile(baseDir, 'results');
P           = 64;      % patch size (pixels)
PURITY      = 0.60;    % a patch needs >= 60 % pixels of one class
N_PER_CLASS = 400;     % patches kept per class (brief asks for >= 200)
SEED        = 2026;    % seed of the hand-written random generator
NAMES6      = {'Building','Land','Road','Vegetation','Water','Unlabeled'};
COL6        = [ 60  16 152;    % #3C1098 Building
               132  41 246;    % #8429F6 Land (unpaved)
               110 193 228;    % #6EC1E4 Road
               254 221  58;    % #FEDD3A Vegetation
               226 169  41;    % #E2A929 Water
               155 155 155];   % #9B9B9B Unlabeled
BUILDING = 1;  ROAD = 3;
CLR_ROAD = COL6(ROAD,:) / 255;   CLR_BLD = COL6(BUILDING,:) / 255;

zipFile = fullfile(baseDir, 'all.zip');
if ~exist(DATA_DIR, 'dir') && ~exist(zipFile, 'file') && ~isempty(DATA_URL)
    fprintf('Downloading the dataset (573 MB) from Kaggle ...\n');
    websave(zipFile, DATA_URL, weboptions('Timeout', 900));   % cloud loading (Task 7)
end
if ~exist(DATA_DIR, 'dir') && exist(zipFile, 'file')
    fprintf('Unzipping %s ...\n', zipFile);
    unzip(zipFile, baseDir);                           % creates all/images and all/masks
end
if ~exist(fullfile(DATA_DIR, 'masks'), 'dir')
    error(['Dataset not found. Put the folder "all" (with images/ and masks/) or all.zip ' ...
           'in %s, or set DATA_URL.'], baseDir);
end
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end
tic;

%% ------------------------------------------- TASK 2a: scan every image/mask
% For each mask: label every pixel by nearest class colour, measure class
% shares, and list every PxP patch whose dominant class reaches PURITY.
nImg      = 203;
pixCount  = zeros(1, 6);           % pixels per class over the whole dataset
pureCount = zeros(1, 6);           % pure patches available per class
exactFrac = zeros(1, nImg);        % share of pixels that match a class colour exactly
imgSize   = zeros(nImg, 2);
valid     = true(1, nImg);
cand      = zeros(0, 5);           % [image, row, col, class, purity]
imgHasCls = false(nImg, 6);        % image contains at least one pixel of class c
for i = 1:nImg
    M = imread(fullfile(DATA_DIR, 'masks', sprintf('mask%04d.png', i)));
    [L, exactFrac(i)] = labelMask(M, COL6);
    imgSize(i, :) = [size(M, 1) size(M, 2)];
    if exactFrac(i) < 0.5          % mask uses no known colour -> corrupted, skip
        valid(i) = false;
        fprintf('  mask %04d skipped: only %.1f %% known colours\n', i, 100 * exactFrac(i));
        continue
    end
    for c = 1:6
        nc = mySum(mySum(L == c));
        pixCount(c) = pixCount(c) + nc;
        imgHasCls(i, c) = nc > 0;
    end
    F = patchFractions(L, P, 6);   % nr x nc x 6 class fractions
    [nr, nc, ~] = size(F);
    for r = 1:nr
        for q = 1:nc
            [best, cls] = maxOf(squeeze(F(r, q, :))');
            if best >= PURITY
                pureCount(cls) = pureCount(cls) + 1;
                if cls == ROAD || cls == BUILDING
                    cand(end + 1, :) = [i, (r - 1) * P + 1, (q - 1) * P + 1, cls, best]; %#ok<AGROW>
                end
            end
        end
    end
    if myMod(i, 25) == 0, fprintf('  scanned %d/%d masks (%.0f s)\n', i, nImg, toc); end
end

%% ------------------------------------------ TASK 2b: balanced, bias-controlled sample
% Roads appear in few images, so BUILDING patches are drawn only from the same
% images that contain ROAD patches. Both classes then share sensors, resolution
% and cities, so a classifier cannot win by recognising the source dataset.
roadIdx = myFind(cand(:, 4) == ROAD)';
hasRoad = false(1, nImg);
hasRoad(cand(roadIdx, 1)) = true;
bldIdx  = myFind(cand(:, 4) == BUILDING & hasRoad(cand(:, 1))')';
fprintf('Road candidates: %d from %d images; matched building candidates: %d\n', ...
        numel(roadIdx), mySum(hasRoad), numel(bldIdx));
if numel(roadIdx) < N_PER_CLASS || numel(bldIdx) < N_PER_CLASS
    error('Not enough pure patches: lower PURITY or P and document the change.');
end
pr  = lcgShuffle(numel(roadIdx), SEED);
pb  = lcgShuffle(numel(bldIdx), SEED + 1);
sel = [roadIdx(pr(1:N_PER_CLASS)), bldIdx(pb(1:N_PER_CLASS))];
N   = numel(sel);
y   = [ones(N_PER_CLASS, 1); zeros(N_PER_CLASS, 1)];   % 1 = road, 0 = building
src = cand(sel, 1);
pur = cand(sel, 5);

% read each needed image once and cut the selected patches
X = zeros(P, P, 3, N, 'uint8');
for i = 1:nImg
    k = myFind(src == i)';
    if isempty(k), continue; end
    I = imread(fullfile(DATA_DIR, 'images', sprintf('image%04d.jpg', i)));
    for j = k
        r0 = cand(sel(j), 2);  c0 = cand(sel(j), 3);
        X(:, :, :, j) = I(r0:r0 + P - 1, c0:c0 + P - 1, :);
    end
end
% Save the refined dataset as image files: 400 road + 400 building 64x64 images
% (Task 2 asks for at least 200 images per class). Roads occur in only 116 of the
% 202 valid scenes, so each class image is a patch, not a whole scene.
REF_DIR = fullfile(RESULTS_DIR, 'refined_dataset');
subDir  = {'building', 'road'};                    % index with y + 1
for k = 1:2
    if ~exist(fullfile(REF_DIR, subDir{k}), 'dir'), mkdir(fullfile(REF_DIR, subDir{k})); end
end
fid = fopen(fullfile(REF_DIR, 'labels.csv'), 'w');
fprintf(fid, 'file,class,source_image,row,col,purity\n');
for j = 1:N
    fn = sprintf('%s/%s_%03d.png', subDir{y(j) + 1}, subDir{y(j) + 1}, j - (1 - y(j)) * N_PER_CLASS);
    imwrite(X(:, :, :, j), fullfile(REF_DIR, fn));
    fprintf(fid, '%s,%s,%d,%d,%d,%.4f\n', fn, subDir{y(j) + 1}, src(j), ...
            cand(sel(j), 2), cand(sel(j), 3), pur(j));
end
fclose(fid);
fprintf('Task 2 done: %d road + %d building images saved to %s (%.0f s)\n', ...
        N_PER_CLASS, N_PER_CLASS, REF_DIR, toc);

%% ------------------------------------------------ TASK 3: feature engineering
[~, featNames] = extractFeatures(X(:, :, :, 1));
nF   = numel(featNames);
FEAT = zeros(N, nF);
for j = 1:N
    FEAT(j, :) = extractFeatures(X(:, :, :, j));
end
fprintf('Task 3 done: %d features x %d patches (%.0f s)\n', nF, N, toc);

%% ------------------------------------------ TASK 4 (part 1): descriptive stats
cls   = {y == 1, y == 0};
cname = {'Road', 'Building'};
S     = zeros(nF, 2, 7);      % mean, median, std, skew, kurt, min, max
D     = zeros(1, nF);         % Cohen's d (road vs building)
for f = 1:nF
    for k = 1:2
        S(f, k, :) = describe(FEAT(cls{k}, f));
    end
    D(f) = cohenD(FEAT(cls{1}, f), FEAT(cls{2}, f));
end
fid = fopen(fullfile(RESULTS_DIR, 'descriptive_stats.csv'), 'w');
fprintf(fid, 'feature,class,mean,median,std,skewness,kurtosis,min,max,cohen_d\n');
for f = 1:nF
    for k = 1:2
        fprintf(fid, '%s,%s,%.5g,%.5g,%.5g,%.4f,%.4f,%.5g,%.5g,%.4f\n', featNames{f}, ...
                cname{k}, S(f, k, 1), S(f, k, 2), S(f, k, 3), S(f, k, 4), S(f, k, 5), ...
                S(f, k, 6), S(f, k, 7), D(f));
    end
end
fclose(fid);

fid = fopen(fullfile(RESULTS_DIR, 'dataset_summary.csv'), 'w');
fprintf(fid, 'class,pixel_share_percent,pure_patches_available\n');
for c = 1:6
    fprintf(fid, '%s,%.3f,%d\n', NAMES6{c}, 100 * pixCount(c) / mySum(pixCount), pureCount(c));
end
fprintf(fid, 'images_total,%d\nimages_valid,%d\nimages_with_any_road_pixel,%d\nimages_with_road_patches,%d\n', ...
        nImg, mySum(valid), mySum(imgHasCls(valid, ROAD)), mySum(hasRoad));
hasBld = false(1, nImg);
hasBld(cand(cand(:, 4) == BUILDING, 1)) = true;
fprintf(fid, 'images_with_building_patches,%d\nsource_images_road_kept,%d\nsource_images_building_kept,%d\n', ...
        mySum(hasBld), mySum(ismemberRow(1:nImg, src(y == 1))), mySum(ismemberRow(1:nImg, src(y == 0))));
fprintf(fid, 'road_candidates,%d\nmatched_building_candidates,%d\nkept_per_class,%d\n', ...
        numel(roadIdx), numel(bldIdx), N_PER_CLASS);
fprintf(fid, 'mean_purity_road,%.4f\nmean_purity_building,%.4f\n', ...
        meanOf(pur(y == 1)), meanOf(pur(y == 0)));
fclose(fid);

%% ------------------------------------------ TASK 4 (part 2): inferential stats
% H0: a feature has the same distribution for road and building patches.
%   Welch t-test      - difference of means, unequal variances, with 95 % CI
%   Mann-Whitney U    - rank test, no normality assumption (many features are skewed)
% nF = 34 features are tested, so significance uses the Bonferroni level 0.05 / 34.
ALPHA = 0.05;  ALPHA_B = ALPHA / nF;
LPT = zeros(1, nF);  LPU = zeros(1, nF);    % log10 p-values (p can underflow double)
fid = fopen(fullfile(RESULTS_DIR, 'inferential_stats.csv'), 'w');
fprintf(fid, ['feature,mean_diff_road_minus_building,ci95_low,ci95_high,welch_t,welch_df,' ...
              'log10_p_t,mannwhitney_U,mannwhitney_z,log10_p_U,rank_biserial_r,significant_bonferroni\n']);
for f = 1:nF
    a = FEAT(cls{1}, f);  b = FEAT(cls{2}, f);
    [t, df, LPT(f), lo, hi] = welchT(a, b);
    [U, z, LPU(f), rb] = mannWhitney(a, b);
    sig = LPT(f) < myLog10(ALPHA_B) && LPU(f) < myLog10(ALPHA_B);
    fprintf(fid, '%s,%.5g,%.5g,%.5g,%.3f,%.1f,%.2f,%.0f,%.3f,%.2f,%.4f,%d\n', featNames{f}, ...
            meanOf(a) - meanOf(b), lo, hi, t, df, LPT(f), U, z, LPU(f), rb, sig);
end
fclose(fid);
fprintf('Inferential: %d of %d features significant in both tests (Bonferroni alpha = %.4f)\n', ...
        mySum(LPT < myLog10(ALPHA_B) & LPU < myLog10(ALPHA_B)), nF, ALPHA_B);

%% ------------------------------------------------------ PCA (hand-coded)
Z = zscore(FEAT);
C = (Z' * Z) / (N - 1);                 % correlation matrix of the features
[V, lam] = jacobiEig(C);
order = descendingOrder(lam);
lam = lam(order);  V = V(:, order);
explained = lam / mySum(lam);
PC = Z * V(:, 1:3);
save(fullfile(RESULTS_DIR, 'features.mat'), 'FEAT', 'featNames', 'y', 'src', 'pur', ...
     'S', 'D', 'LPT', 'LPU', 'C', 'lam', 'V', 'explained', 'P', 'PURITY', 'SEED');
fprintf('PCA: PC1-3 explain %.1f %% of variance\n', 100 * mySum(explained(1:3)));

%% ------------------------------- XAI preview of Task 6 (white-box prototype)
% Week 1: explanations come from a transparent ("white box") model plus two XAI
% methods: feature importance and counterfactual explanations.
% Model: L2-regularised logistic regression on standardised features,
%   P(road | x) = 1 / (1 + exp(-(b + w'z))),  z = (x - mu_train) / sd_train.
% Validation: 5-fold cross-validation GROUPED BY SOURCE IMAGE, so patches of one
% image are never in both training and test folds (no leakage between neighbours).
K_FOLD = 5;  LAMBDA = 0.05;
imgs   = uniqueInts(src);
po     = lcgShuffle(numel(imgs), SEED + 2);
foldOf = zeros(N, 1);
for k = 1:numel(imgs)
    foldOf(src == imgs(po(k))) = myMod(k - 1, K_FOLD) + 1;      % round-robin images
end
CM  = zeros(K_FOLD, 4);              % TP, FP, TN, FN per fold
PIm = zeros(K_FOLD, nF);             % permutation importance per fold
pOut = zeros(N, 1);                  % out-of-fold P(road)
for k = 1:K_FOLD
    tr = foldOf ~= k;  te = foldOf == k;
    mdl = trainLogReg(FEAT(tr, :), y(tr), LAMBDA);
    pOut(te) = predictLogReg(mdl, FEAT(te, :));
    yh = pOut(te) >= 0.5;  yt = y(te) == 1;
    CM(k, :) = [mySum(yh & yt), mySum(yh & ~yt), mySum(~yh & ~yt), mySum(~yh & yt)];
    acc0 = meanOf(double(yh == yt));
    Xte = FEAT(te, :);
    for f = 1:nF                     % importance = accuracy lost when f is shuffled
        Xp = Xte;  Xp(:, f) = Xte(lcgShuffle(mySum(te), SEED + 10 * k + f), f);
        PIm(k, f) = acc0 - meanOf(double((predictLogReg(mdl, Xp) >= 0.5) == yt));
    end
end
T = mySum(CM, 1);
acc = (T(1) + T(3)) / N;  prec = T(1) / (T(1) + T(2));  rec = T(1) / (T(1) + T(4));
spec = T(3) / (T(3) + T(2));  f1 = 2 * prec * rec / (prec + rec);
% Inferential statistics on the model's performance (Task 4 applied to Task 6):
%   ROC curve and AUC from the out-of-fold probabilities, with a Hanley-McNeil 95 % CI;
%   Wilson 95 % CI for accuracy and an exact one-sided binomial test of H0: accuracy = 0.5.
[rocF, rocT, auc] = rocCurve(pOut, y);
[aucLo, aucHi]   = aucCI(auc, mySum(y == 1), mySum(y == 0));
[accLo, accHi]   = wilsonCI(T(1) + T(3), N);
lpBin            = log10BinomTail(T(1) + T(3), N);
fid = fopen(fullfile(RESULTS_DIR, 'xai_cv_metrics.csv'), 'w');
fprintf(fid, 'fold,test_images,test_patches,TP,FP,TN,FN,accuracy\n');
for k = 1:K_FOLD
    fprintf(fid, '%d,%d,%d,%d,%d,%d,%d,%.4f\n', k, numel(uniqueInts(src(foldOf == k))), ...
            mySum(foldOf == k), CM(k, :), (CM(k, 1) + CM(k, 3)) / mySum(CM(k, :)));
end
fprintf(fid, 'pooled,%d,%d,%d,%d,%d,%d,%.4f\n', numel(imgs), N, T, acc);
fprintf(fid, 'precision,%.4f\nrecall,%.4f\nspecificity,%.4f\nF1,%.4f\n', prec, rec, spec, f1);
fprintf(fid, 'AUC,%.4f\nAUC_CI95_low,%.4f\nAUC_CI95_high,%.4f\n', auc, aucLo, aucHi);
fprintf(fid, 'accuracy_CI95_low,%.4f\naccuracy_CI95_high,%.4f\nbinomial_log10_p_vs_0.5,%.1f\n', ...
        accLo, accHi, lpBin);
fclose(fid);
fprintf('XAI prototype: grouped %d-fold accuracy %.3f (95%% CI %.3f-%.3f), F1 %.3f, AUC %.3f (%.3f-%.3f)\n', ...
        K_FOLD, acc, accLo, accHi, f1, auc, aucLo, aucHi);
fprintf('Binomial test vs chance (0.5): p = 1e%.1f\n', lpBin);

% Reduced feature vector (Task 3 -> Task 6): the same model and folds, but fed the
% first k principal components instead of all nF features. Standardisation and PCA
% are fitted on the training folds only, so the test images never shape the axes.
K_PCS = [3 5 10];
PCAcmp = zeros(numel(K_PCS) + 1, 3);          % k, accuracy, AUC  (last row = all features)
for r = 1:numel(K_PCS)
    pk = zeros(N, 1);
    for k = 1:K_FOLD
        tr = foldOf ~= k;  te = foldOf == k;
        [Ztr, Zte] = standardiseTrainTest(FEAT(tr, :), FEAT(te, :));
        [Vk, lk] = jacobiEig((Ztr' * Ztr) / (mySum(tr) - 1));
        ok = descendingOrder(lk);  Vk = Vk(:, ok(1:K_PCS(r)));
        m  = trainLogReg(Ztr * Vk, y(tr), LAMBDA);
        pk(te) = predictLogReg(m, Zte * Vk);
    end
    [~, ~, aucK] = rocCurve(pk, y);
    PCAcmp(r, :) = [K_PCS(r), meanOf(double((pk >= 0.5) == (y == 1))), aucK];
end
PCAcmp(end, :) = [nF, acc, auc];
fid = fopen(fullfile(RESULTS_DIR, 'xai_pca_comparison.csv'), 'w');
fprintf(fid, 'model_input,dimensions,accuracy,AUC\n');
for r = 1:numel(K_PCS)
    fprintf(fid, 'first_%d_PCs,%d,%.4f,%.4f\n', K_PCS(r), PCAcmp(r, :));
end
fprintf(fid, 'all_features,%d,%.4f,%.4f\n', PCAcmp(end, :));
fclose(fid);
for r = 1:numel(K_PCS)
    fprintf('  model on first %2d PCs: accuracy %.3f, AUC %.3f\n', PCAcmp(r, :));
end

% Global explanation: one model on all patches; its standardised weights are
% directly readable (white box), next to the model-agnostic permutation importance.
mdl = trainLogReg(FEAT, y, LAMBDA);
PImean = mySum(PIm, 1) / K_FOLD;
fid = fopen(fullfile(RESULTS_DIR, 'xai_feature_importance.csv'), 'w');
fprintf(fid, 'feature,std_weight,odds_ratio_per_sd,perm_importance_mean,perm_importance_min,perm_importance_max\n');
for f = 1:nF
    fprintf(fid, '%s,%.4f,%.4f,%.4f,%.4f,%.4f\n', featNames{f}, mdl.w(f), myExp(mdl.w(f)), ...
            PImean(f), -maxOf(-PIm(:, f)'), maxOf(PIm(:, f)'));
end
fclose(fid);

% Local explanation (counterfactual): for a linear model the smallest change of ONE
% feature j that moves a patch to the decision boundary is  dz_j = -s / w_j,
% s = b + w'z  (in standard deviations); x_j changes by dz_j * sd_j in real units.
% It is shown for the most uncertain correctly classified patch of each class.
s = mdl.b + ((FEAT - mdl.mu) ./ mdl.sd) * mdl.w;
CF = zeros(2, 4);                    % patch index, feature, change (real units), P(road)
for k = 1:2
    cand_k = myFind(y == 2 - k & ((s > 0) == (y == 1)));      % k=1 road, k=2 building
    [~, j0] = maxOf(-myAbs(s(cand_k))');  j = cand_k(j0);
    dz = -s(j) ./ mdl.w';  dz(myAbs(mdl.w') < 1e-3) = inf;  % ignore near-zero weights
    [~, fbest] = maxOf(-myAbs(dz));
    CF(k, :) = [j, fbest, 1.05 * dz(fbest) * mdl.sd(fbest), 1 / (1 + myExp(-s(j)))];
end
fid = fopen(fullfile(RESULTS_DIR, 'xai_counterfactuals.csv'), 'w');
fprintf(fid, 'patch,true_class,p_road,feature,value,counterfactual_value,change\n');
for k = 1:2
    j = CF(k, 1);  f = CF(k, 2);
    fprintf(fid, '%d,%s,%.3f,%s,%.5g,%.5g,%.5g\n', j, cname{k}, CF(k, 4), featNames{f}, ...
            FEAT(j, f), FEAT(j, f) + CF(k, 3), CF(k, 3));
end
fclose(fid);
save(fullfile(RESULTS_DIR, 'features.mat'), 'foldOf', 'pOut', 'PIm', 'mdl', 'CF', 'PCAcmp', '-append');

%% ------------------------------------------ TASK 6: model comparison
% The logistic regression above is compared with seven other classic (non-neural)
% classifiers on exactly the same grouped folds, so every model is trained and
% tested on the same images. The seven are written by hand further down:
%   k-NN        - share of road patches among the k nearest standardised patches
%   Naive Bayes - Gaussian density per class and feature, features assumed independent
%   LDA         - Fisher's linear discriminant with a shrunk pooled covariance
%   Linear SVM  - hinge loss + L2 penalty, sub-gradient descent
%   RBF SVM     - the same loss with a Gaussian kernel (kernel Pegasos)
%   Tree        - CART, splits chosen by Gini impurity
%   Forest      - 200 CART trees on bootstrap samples, 6 random features per split
% The k of k-NN and the depth of the tree are tuned INSIDE each training set by
% nested grouped CV (each of the four training folds is held out once), so the test
% fold never helps choose them.
K_GRID     = [1 3 5 7 9 15 21 31];
DEPTH_GRID = [1 2 3 4 6 8];
MODELS = {'Logistic regression', 'k-NN', 'Gaussian naive Bayes', 'LDA (shrinkage)', ...
          'Linear SVM', 'Decision tree (CART)', 'RBF-kernel SVM', 'Random forest'};
nM     = numel(MODELS);
SC     = zeros(N, nM);              % out-of-fold scores, higher = more road-like
TH     = [0.5 0.5 0.5 0 0 0.5 0 0.5];   % decision threshold on each model's score
chosen = zeros(K_FOLD, 2);          % tuned k (k-NN) and depth (tree) per outer fold
SC(:, 1) = pOut;
for k = 1:K_FOLD
    tr = foldOf ~= k;  te = foldOf == k;
    [Ztr, Zte] = standardiseTrainTest(FEAT(tr, :), FEAT(te, :));
    ytr = y(tr);
    chosen(k, 1) = tuneInner('knn',  Ztr, ytr, foldOf(tr), K_GRID);
    chosen(k, 2) = tuneInner('tree', Ztr, ytr, foldOf(tr), DEPTH_GRID);
    SC(te, 2) = knnScores(Ztr, ytr, Zte, chosen(k, 1));
    SC(te, 3) = nbScore(Ztr, ytr, Zte);
    SC(te, 4) = ldaScore(Ztr, ytr, Zte);
    SC(te, 5) = svmScore(Ztr, ytr, Zte);
    SC(te, 6) = predictTree(growTree(Ztr, ytr, chosen(k, 2), 5), Zte, chosen(k, 2));
    SC(te, 7) = rbfSvmScore(Ztr, ytr, Zte);
    SC(te, 8) = forestScore(Ztr, ytr, Zte, 200, 6, SEED + 10 + k);
end
% Metrics per model, a Wilson 95 % CI for accuracy, and an exact McNemar test of
% H0: "this model and logistic regression make errors equally often" (paired, same patches).
R = zeros(nM, 8);                   % acc, CI low, CI high, precision, recall, specificity, F1, AUC
MC = zeros(nM, 3);                  % McNemar: b (only LR right), c (only model right), log10 p
okLR = (SC(:, 1) >= TH(1)) == (y == 1);
for m = 1:nM
    R(m, :) = binaryMetrics(SC(:, m), TH(m), y);
    okM = (SC(:, m) >= TH(m)) == (y == 1);
    MC(m, :) = [mySum(okLR & ~okM), mySum(~okLR & okM), mcnemarExact(mySum(okLR & ~okM), mySum(~okLR & okM))];
end
fid = fopen(fullfile(RESULTS_DIR, 'model_comparison.csv'), 'w');
fprintf(fid, ['model,tuned_value_per_fold,accuracy,acc_ci95_low,acc_ci95_high,precision,recall,' ...
              'specificity,F1,AUC,mcnemar_only_LR_right,mcnemar_only_model_right,mcnemar_log10_p\n']);
for m = 1:nM
    tv = '-';
    if m == 2, tv = sprintf('k=%s', sprintf('%d ', chosen(:, 1))); end
    if m == 6, tv = sprintf('depth=%s', sprintf('%d ', chosen(:, 2))); end
    fprintf(fid, '%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%.2f\n', ...
            MODELS{m}, strtrim(tv), R(m, :), MC(m, :));
end
fclose(fid);
fprintf('Task 6 model comparison (grouped %d-fold CV, nested tuning):\n', K_FOLD);
for m = 1:nM
    fprintf('  %-22s accuracy %.3f (%.3f-%.3f)  F1 %.3f  AUC %.3f  McNemar vs LR p %s\n', ...
            MODELS{m}, R(m, 1:3), R(m, 7), R(m, 8), pText(MC(m, 3)));
end
save(fullfile(RESULTS_DIR, 'features.mat'), 'SC', 'R', 'MC', 'chosen', 'MODELS', '-append');

%% ------------------------------------------- TASK 5 (part 1): visualisation
plotDataset(DATA_DIR, RESULTS_DIR, cand, sel, y, COL6, NAMES6, pixCount, pureCount, P);
plotSamples(X, y, RESULTS_DIR);
plotHistograms(X, y, RESULTS_DIR, CLR_ROAD, CLR_BLD);
plotPipeline(X, y, RESULTS_DIR);
plotViolins(FEAT, y, featNames, D, LPT, LPU, RESULTS_DIR, CLR_ROAD, CLR_BLD);
plotViolinsOneAxis(FEAT, y, featNames, D, LPT, LPU, RESULTS_DIR, CLR_ROAD, CLR_BLD);
plotCorrelation(C, featNames, RESULTS_DIR);
plotPCA(PC, explained, y, pOut, RESULTS_DIR, CLR_ROAD, CLR_BLD);
plotXAI(X, FEAT, featNames, mdl, PIm, CF, acc, RESULTS_DIR, CLR_ROAD, CLR_BLD);
plotROC(rocF, rocT, auc, aucLo, aucHi, spec, rec, RESULTS_DIR);
plotModels(MODELS, R, SC, y, RESULTS_DIR);
fprintf('All done in %.0f s. Results in %s\n', toc, RESULTS_DIR);
end

%% =====================================================================
%%                        TASK 2 helper functions
%% =====================================================================

function tf = ismemberRow(v, list)
% tf(k) is true when v(k) appears in list (hand-written ismember)
tf = false(1, numel(v));
for k = 1:numel(list)
    tf = tf | (v == list(k));
end
end
function [L, exact] = labelMask(M, COL)
% Label each pixel with the class whose colour is nearest in RGB space:
%   L(x,y) = argmin_c (R-Rc)^2 + (G-Gc)^2 + (B-Bc)^2
% Rows are processed in blocks of 512 to keep memory small for 6000x6000 masks.
[H, W, ~] = size(M);
L = zeros(H, W, 'uint8');
nExact = 0;
for r0 = 1:512:H
    r1  = myMin(r0 + 511, H);
    blk = double(M(r0:r1, :, :));
    best = inf(r1 - r0 + 1, W);
    lab  = zeros(r1 - r0 + 1, W);
    for k = 1:size(COL, 1)
        d = (blk(:, :, 1) - COL(k, 1)).^2 + (blk(:, :, 2) - COL(k, 2)).^2 + ...
            (blk(:, :, 3) - COL(k, 3)).^2;
        better = d < best;               % pixels closer to class k than before
        best(better) = d(better);
        lab(better)  = k;
    end
    nExact = nExact + mySum(best(:) == 0);
    L(r0:r1, :) = uint8(lab);
end
exact = nExact / (H * W);
end

function F = patchFractions(L, P, nC)
% Fraction of pixels of each class inside every non-overlapping PxP patch.
% The image is cropped to a multiple of P, then reshaped so each patch becomes
% one slice; summing a slice counts its pixels (no loop over pixels needed).
nr = myFloor(size(L, 1) / P);  nc = myFloor(size(L, 2) / P);
L  = L(1:nr * P, 1:nc * P);
F  = zeros(nr, nc, nC);
for c = 1:nC
    R = reshape(double(L == c), P, nr, P, nc);
    F(:, :, c) = reshape(mySum(mySum(R, 1), 3), nr, nc) / (P * P);
end
end

function p = lcgShuffle(n, seed)
% Fisher-Yates shuffle driven by a linear congruential generator
%   s_{k+1} = (a s_k + c) mod m,  a = 1103515245, c = 12345, m = 2^31
p = 1:n;  s = seed;
for i = n:-1:2
    s = myMod(1103515245 * s + 12345, 2^31);
    j = myFloor(s / 2^31 * i) + 1;              % uniform index in 1..i
    t = p(i);  p(i) = p(j);  p(j) = t;
end
end

%% =====================================================================
%%                     TASK 3: feature engineering
%% =====================================================================
function [f, names] = extractFeatures(rgb)
% 34 features in nine families. All maths is written out in the report.
LAP_ZC_STEP = 0.02;     % a sign change counts as a Laplacian zero crossing only if the jump exceeds this
I = double(rgb) / 255;                                  % RGB in [0,1]
Y = 0.299 * I(:, :, 1) + 0.587 * I(:, :, 2) + 0.114 * I(:, :, 3);   % luminance (ITU-R BT.601)
Ys = preprocess(Y);                                     % Week 3: median, then Gaussian

% (1) colour statistics -------------------------------------------- 6
cs = zeros(1, 6);
for b = 1:3
    v = I(:, :, b);  v = v(:);
    cs(b)     = meanOf(v);
    cs(b + 3) = mySqrt(meanOf((v - cs(b)).^2));
end
% (2) luminance moments -------------------------------------------- 4
m = moments(Ys(:));                                     % mean, std, skewness, kurtosis
% (3) edges (Sobel) ------------------------------------------------- 4
Gx  = conv2same(Ys, [-1 0 1; -2 0 2; -1 0 1]);
Gy  = conv2same(Ys, [-1 -2 -1; 0 0 0; 1 2 1]);
mag = mySqrt(Gx.^2 + Gy.^2);
th  = myAtan2(Gy, Gx);                                    % gradient direction
oh  = orientationHistogram(mag, th);                    % 4 bins, magnitude-weighted
ed  = [meanOf(mag(:)), meanOf(double(mag(:) > 0.25)), maxOf(oh), entropyOf(oh) / myLog(4)];
% (4) Otsu threshold + morphology + shape -------------------------- 4
t   = otsuThreshold(Ys);
B   = Ys > t;
sh1 = shapeOf(openBinary(B));                           % bright phase
sh2 = shapeOf(openBinary(~B));                          % dark phase
if sh1(2) >= sh2(2), sh = sh1; else, sh = sh2; end      % keep the more elongated phase
mo  = [t, sh];                                          % threshold, area, elongation, compactness
% (5) texture (GLCM) ------------------------------------------------ 5
gl  = glcmFeatures(Ys, 8);
% (6) frequency (2-D DFT) ------------------------------------------- 2
fr  = fourierFeatures(Ys);
% (7) taught thresholding and grey-scale morphology (Weeks 2 and 5) - 4
[t1, t2] = otsu2Thresholds(Ys);                         % multi-level Otsu, 3 classes
op  = greyDilate3(greyDilate3(greyErode3(greyErode3(Ys))));   % opening, 5x5 square
cl  = greyErode3(greyErode3(greyDilate3(greyDilate3(Ys))));   % closing, 5x5 square
tm  = [kapurThreshold(Ys), meanOf(double(Ys(:) > t1 & Ys(:) <= t2)), ...
       meanOf(Ys(:) - op(:)), meanOf(cl(:) - Ys(:))];   % white / black top-hat means

% (8) Laplacian (second derivative) ----------------------------------- 2
%   L = d2Y/dx2 + d2Y/dy2 with the 4-neighbour kernel [0 1 0; 1 -4 1; 0 1 0]
Lp  = conv2same(Ys, [0 1 0; 1 -4 1; 0 1 0]);
zc  = (Lp(:, 1:end - 1) .* Lp(:, 2:end) < 0) & (myAbs(Lp(:, 1:end - 1) - Lp(:, 2:end)) > LAP_ZC_STEP);
zr  = (Lp(1:end - 1, :) .* Lp(2:end, :) < 0) & (myAbs(Lp(1:end - 1, :) - Lp(2:end, :)) > LAP_ZC_STEP);
la  = [meanOf(myAbs(Lp(:))), (mySum(zc(:)) + mySum(zr(:))) / (numel(zc) + numel(zr))];
% (9) histogram of oriented gradients (HOG), 2x2 cells x 9 bins ------- 3
H   = hogCells(mag, th, 2, 9);                          % 4 x 9, block-normalised (L2-Hys)
hp  = mySum(H, 1) / (mySum(H(:)) + EPS());                    % pooled histogram
cellMax = zeros(1, 4);  cosSim = [];
for c = 1:4
    cellMax(c) = maxOf(H(c, :)) / (mySum(H(c, :)) + EPS());
    for d = c + 1:4
        cosSim(end + 1) = (H(c, :) * H(d, :)') / (myNorm(H(c, :)) * myNorm(H(d, :)) + EPS()); %#ok<AGROW>
    end
end
hg  = [entropyOf(hp) / myLog(9), meanOf(cellMax), meanOf(cosSim)];

f = [cs, m, ed, mo, gl, fr, tm, la, hg];
names = {'meanR','meanG','meanB','stdR','stdG','stdB', ...
         'meanY','stdY','skewY','kurtY', ...
         'gradMean','edgeDensity','orientDominance','orientEntropy', ...
         'otsuT','phaseArea','elongation','compactness', ...
         'glcmContrast','glcmEnergy','glcmHomogeneity','glcmEntropy','glcmCorrelation', ...
         'lowFreqRatio','spectralAnisotropy', ...
         'kapurT','otsu3MidShare','whiteTopHat','blackTopHat', ...
         'lapAbsMean','lapZeroCross', ...
         'hogEntropy','hogCellDominance','hogCellAgreement'};
end

function H = hogCells(mag, th, nCell, nBin)
% Histogram of oriented gradients (Dalal and Triggs, 2005) on an nCell x nCell grid.
% Unsigned direction in [0, pi), nBin bins; each pixel votes its gradient magnitude,
% split linearly between the two nearest bin centres. The cells form one block,
% normalised with L2-Hys: v = v/|v|, clip at 0.2, renormalise.
a  = myMod(th, PI());  bw = PI() / nBin;
cs = size(mag, 1) / nCell;
H  = zeros(nCell * nCell, nBin);
for r = 1:nCell
    for c = 1:nCell
        ri = (r - 1) * cs + 1:r * cs;  ci = (c - 1) * cs + 1:c * cs;
        am = a(ri, ci);  mm = mag(ri, ci);
        pos = am / bw - 0.5;  b0 = myFloor(pos);  fr = pos - b0;
        b0 = myMod(b0, nBin) + 1;  b1 = myMod(b0, nBin) + 1;
        h = zeros(1, nBin);
        for b = 1:nBin
            h(b) = mySum(mm(b0 == b) .* (1 - fr(b0 == b))) + mySum(mm(b1 == b) .* fr(b1 == b));
        end
        H((r - 1) * nCell + c, :) = h;
    end
end
v = H(:);  v = v / mySqrt(mySum(v.^2) + 1e-12);
v(v > 0.2) = 0.2;  v = v / mySqrt(mySum(v.^2) + 1e-12);
H = reshape(v, size(H));
end

function [Ztr, Zte] = standardiseTrainTest(Ftr, Fte)
% z-scores with the training mean and SD, applied to both sets (no test leakage)
n = size(Ftr, 1);  mu = mySum(Ftr, 1) / n;
sd = mySqrt(mySum((Ftr - mu).^2, 1) / (n - 1));  sd(sd < 1e-12) = 1e-12;
Ztr = (Ftr - mu) ./ sd;  Zte = (Fte - mu) ./ sd;
end

function Ys = preprocess(Y)
% Noise reduction exactly as in the Week 3 and Week 5 labs:
%   3x3 median filter (symmetric padding) removes JPEG speckle but keeps edges,
%   then the 3x3 Gaussian with sigma = sqrt(3),  K(i,j) ~ exp(-(i^2 + j^2) / (2 sigma^2)),
%   normalised to sum 1 and applied with replicate padding (Week 4).
sigma = mySqrt(3);
K = zeros(3);
for i = -1:1
    for j = -1:1
        K(i + 2, j + 2) = myExp(-(i^2 + j^2) / (2 * sigma^2));
    end
end
K  = K / mySum(K(:));
Ys = conv2same(median3(Y), K);
end

function M = median3(A)
% 3x3 median filter. The 9 neighbours of every pixel are stacked as 9 images and
% sorted with element-wise min/max compare-swaps (bubble sort); the 5th is the median.
[h, w] = size(A);
Ap = [A(1, :); A; A(end, :)];  Ap = [Ap(:, 1), Ap, Ap(:, end)];   % symmetric pad, width 1
S = zeros(h, w, 9);  k = 0;
for s = 0:2
    for t = 0:2
        k = k + 1;  S(:, :, k) = Ap(1 + s:h + s, 1 + t:w + t);
    end
end
for i = 1:8
    for j = 1:9 - i
        lo = myMin(S(:, :, j), S(:, :, j + 1));  hi = myMax(S(:, :, j), S(:, :, j + 1));
        S(:, :, j) = lo;  S(:, :, j + 1) = hi;
    end
end
M = S(:, :, 5);
end

function E = greyErode3(A)
% Grey-scale erosion, 3x3 flat square: minimum of the neighbourhood (Week 2).
% Applying it twice equals erosion with a 5x5 square.
[h, w] = size(A);
Ap = [A(1, :); A; A(end, :)];  Ap = [Ap(:, 1), Ap, Ap(:, end)];
E = inf(h, w);
for s = 0:2
    for t = 0:2
        E = myMin(E, Ap(1 + s:h + s, 1 + t:w + t));
    end
end
end

function D = greyDilate3(A)
% Grey-scale dilation, 3x3 flat square: maximum of the neighbourhood (Week 2).
[h, w] = size(A);
Ap = [A(1, :); A; A(end, :)];  Ap = [Ap(:, 1), Ap, Ap(:, end)];
D = -inf(h, w);
for s = 0:2
    for t = 0:2
        D = myMax(D, Ap(1 + s:h + s, 1 + t:w + t));
    end
end
end

function p = greyHistogram(Y)
% Normalised 256-bin histogram of an image with values in [0,1].
q = myFloor(Y(:) * 255);  n = numel(q);
p = zeros(1, 256);
for g = 0:255
    p(g + 1) = mySum(q == g) / n;
end
end

function t = kapurThreshold(Y)
% Kapur maximum-entropy threshold (Week 5): choose k that maximises
%   H_b(k) + H_f(k),  H_b = -sum_{g<=k} (p_g/P_b) ln(p_g/P_b),  H_f likewise above k.
p = greyHistogram(Y);  best = -inf;  t = 0.5;
for k = 1:255
    Pb = mySum(p(1:k));  Pf = 1 - Pb;
    if Pb < 1e-9 || Pf < 1e-9, continue; end
    H = entropyOf(p(1:k) / Pb) + entropyOf(p(k + 1:end) / Pf);
    if H > best, best = H; t = (k - 0.5) / 255; end
end
end

function [t1, t2] = otsu2Thresholds(Y)
% Two-level Otsu (Week 5 multi-level thresholding): over every pair k1 < k2 maximise
% the between-class variance  sigma_B^2 = sum_c w_c (mu_c - mu_T)^2  of the 3 classes.
% Cumulative sums give w_c and mu_c for all pairs at once.
p = greyHistogram(Y);  lv = 0:255;
P = zeros(1, 256);  Sm = zeros(1, 256);  a = 0;  b = 0;
for g = 1:256
    a = a + p(g);  b = b + lv(g) * p(g);  P(g) = a;  Sm(g) = b;
end
muT = Sm(end);
[K2, K1] = myMeshgrid(1:255, 1:255);                     % k1 = last level of class 1
w0 = P(K1);         w1 = P(K2) - P(K1);       w2 = 1 - P(K2);
s0 = Sm(K1);        s1 = Sm(K2) - Sm(K1);     s2 = muT - Sm(K2);
sb = s0.^2 ./ myMax(w0, 1e-12) + s1.^2 ./ myMax(w1, 1e-12) + s2.^2 ./ myMax(w2, 1e-12) - muT^2;
sb(K2 <= K1 | w0 < 1e-9 | w1 < 1e-9 | w2 < 1e-9) = -inf;
k = argmaxLarge(sb(:)');
t1 = (K1(k) - 0.5) / 255;  t2 = (K2(k) - 0.5) / 255;
end

function k = argmaxLarge(v)
% Index of the largest element of a long vector: halve it with element-wise
% max of pairs until one value is left (fast where a scalar loop is slow).
m = v;
while numel(m) > 1
    if myMod(numel(m), 2) == 1, m = [m, -inf]; end
    m = myMax(m(1:2:end), m(2:2:end));
end
k = myFind(v == m, 1);
end

function O = conv2same(A, K)
% 2-D convolution, output the same size as A, replicate padding (Week 4):
%   O(x,y) = sum_s sum_t K(s,t) * A(x - s, y - t)
[h, w] = size(A);  k = (size(K, 1) - 1) / 2;
Ap = [myRepmat(A(1, :), k, 1); A; myRepmat(A(end, :), k, 1)];
Ap = [myRepmat(Ap(:, 1), 1, k), Ap, myRepmat(Ap(:, end), 1, k)];
Kf = K(end:-1:1, end:-1:1);                    % flip kernel = true convolution
O  = zeros(h, w);
for s = 1:size(K, 1)
    for t = 1:size(K, 2)
        O = O + Kf(s, t) * Ap(s:s + h - 1, t:t + w - 1);
    end
end
end

function m = moments(v)
% mean, standard deviation, skewness, kurtosis (population moments)
mu = meanOf(v);  d = v - mu;
s  = mySqrt(meanOf(d.^2));
if s < 1e-12, m = [mu 0 0 3]; return; end
m  = [mu, s, meanOf(d.^3) / s^3, meanOf(d.^4) / s^4];
end

function h = orientationHistogram(mag, th)
% Edge orientation folded to [0, pi) and binned at 0, 45, 90, 135 degrees,
% each pixel weighted by its gradient magnitude; histogram sums to 1.
a  = myMod(th, PI());
bn = myMod(myFloor((a + PI() / 8) / (PI() / 4)), 4) + 1;
h  = zeros(1, 4);
for b = 1:4
    h(b) = mySum(mag(bn == b));
end
h = h / (mySum(h) + EPS());
end

function t = otsuThreshold(Y)
% Otsu: choose t that minimises the within-class variance (Week 5)
%   sigma_W^2(t) = w_b(t) sigma_b^2(t) + w_f(t) sigma_f^2(t)
q = myFloor(Y(:) * 255);  n = numel(q);
p = zeros(1, 256);
for g = 0:255
    p(g + 1) = mySum(q == g) / n;
end
lv = 0:255;  bestW = inf;  t = 0.5;
for k = 1:255
    wb = mySum(p(1:k));  wf = 1 - wb;
    if wb < 1e-9 || wf < 1e-9, continue; end
    mb = mySum(lv(1:k) .* p(1:k)) / wb;
    mf = mySum(lv(k + 1:end) .* p(k + 1:end)) / wf;
    vb = mySum(((lv(1:k) - mb).^2) .* p(1:k)) / wb;
    vf = mySum(((lv(k + 1:end) - mf).^2) .* p(k + 1:end)) / wf;
    sw = wb * vb + wf * vf;
    if sw < bestW, bestW = sw; t = (k - 0.5) / 255; end
end
end

function B = openBinary(B)
% Morphological opening = erosion then dilation with a 3x3 square (Week 2)
B = dilate3(erode3(B));
end

function E = erode3(B)
[h, w] = size(B);  Bp = true(h + 2, w + 2);  Bp(2:h + 1, 2:w + 1) = B;
E = true(h, w);
for s = 0:2
    for t = 0:2
        E = E & Bp(1 + s:h + s, 1 + t:w + t);
    end
end
end

function Dl = dilate3(B)
[h, w] = size(B);  Bp = false(h + 2, w + 2);  Bp(2:h + 1, 2:w + 1) = B;
Dl = false(h, w);
for s = 0:2
    for t = 0:2
        Dl = Dl | Bp(1 + s:h + s, 1 + t:w + t);
    end
end
end

function sh = shapeOf(B)
% Region shape of a binary phase:
%   area        A = share of pixels in the phase
%   elongation  = sqrt(l1 / l2), l1 >= l2 eigenvalues of the 2x2 coordinate covariance
%   compactness = 4 pi A_px / Perim^2 (1 = disc, small = thin or ragged)
[r, c] = myFind2(B);
n = numel(r);
if n < 3, sh = [0, 1, 0]; return; end
mr = meanOf(r);  mc = meanOf(c);
a = meanOf((r - mr).^2);  b = meanOf((r - mr) .* (c - mc));  d = meanOf((c - mc).^2);
root = mySqrt(((a - d) / 2)^2 + b^2);
l1 = (a + d) / 2 + root;  l2 = (a + d) / 2 - root;
el = mySqrt(l1 / myMax(l2, 1e-6));
perim = mySum(mySum(B & ~erode3(B)));                         % boundary pixels
sh = [n / numel(B), myMin(el, 50), 4 * PI() * n / myMax(perim, 1)^2];
end

function g = glcmFeatures(Y, Lv)
% Grey-level co-occurrence matrix, Lv levels, offsets 0, 45, 90, 135 degrees,
% made symmetric and normalised; then Haralick-style statistics.
q = myMin(myFloor(Y * Lv), Lv - 1) + 1;
[h, w] = size(q);
offs = [0 1; -1 1; -1 0; -1 -1];
Pm = zeros(Lv);
for o = 1:4
    dr = offs(o, 1);  dc = offs(o, 2);
    r1 = myMax(1, 1 - dr):myMin(h, h - dr);  c1 = myMax(1, 1 - dc):myMin(w, w - dc);
    a  = q(r1, c1);  b = q(r1 + dr, c1 + dc);
    idx = (a(:) - 1) * Lv + b(:);
    for v = 1:Lv * Lv
        Pm(v) = Pm(v) + mySum(idx == v);
    end
end
Pm = Pm + Pm';  Pm = Pm / mySum(Pm(:));
[J, Ii] = myMeshgrid(1:Lv, 1:Lv);   % note: Pm(v) fills column-major, symmetric anyway
mi = mySum(mySum(Ii .* Pm));  mj = mySum(mySum(J .* Pm));
si = mySqrt(mySum(mySum((Ii - mi).^2 .* Pm)));  sj = mySqrt(mySum(mySum((J - mj).^2 .* Pm)));
nz = Pm(Pm > 0);
g = [mySum(mySum((Ii - J).^2 .* Pm)), ...                       % contrast
     mySum(mySum(Pm.^2)), ...                                   % energy
     mySum(mySum(Pm ./ (1 + myAbs(Ii - J)))), ...                 % homogeneity
     -mySum(nz .* myLog(nz)), ...                               % entropy
     mySum(mySum((Ii - mi) .* (J - mj) .* Pm)) / myMax(si * sj, 1e-12)];   % correlation
end

function Sp = dftPower(Y)
% |F|^2 of the 2-D DFT F = W (Y - mean) W.',  W(k,n) = e^(-2 pi i k n / N).
% With C = cos(2 pi k n / N) and S = sin(2 pi k n / N), W = C - iS, so
%   real(F) = C Y C' - S Y S'  and  imag(F) = -(C Y S' + S Y C'),
% and every step stays in real numbers. k n is reduced mod N first, so each
% angle lies in [0, 2 pi).
N = size(Y, 1);  n = 0:N - 1;
[Sn, Cn] = mySinCos(2 * PI() * myMod((n') * n, N) / N);
Y0 = Y - meanOf(Y(:));
Fr = Cn * Y0 * Cn.' - Sn * Y0 * Sn.';
Fi = -(Cn * Y0 * Sn.' + Sn * Y0 * Cn.');
Sp = Fr .^ 2 + Fi .^ 2;
end

function fr = fourierFeatures(Y)
% lowFreqRatio       = energy with radius <= N/8 / total energy (DC removed)
% spectralAnisotropy = largest share of energy in one 45-degree direction sector
N = size(Y, 1);  n = 0:N - 1;
Sp = dftPower(Y);
u = myMod(n + N / 2, N) - N / 2;                 % signed frequency index
[U, V] = myMeshgrid(u, u);
rad = mySqrt(U.^2 + V.^2);
tot = mySum(Sp(:)) + EPS();
ang = myMod(myAtan2(V, U), PI());
sec = myMod(myFloor((ang + PI() / 8) / (PI() / 4)), 4) + 1;
e = zeros(1, 4);
for b = 1:4
    e(b) = mySum(Sp(sec == b & rad > 0));
end
fr = [mySum(Sp(rad <= N / 8)) / tot, maxOf(e / (mySum(e) + EPS()))];
end

%% =====================================================================
%%                   statistics helpers (hand-written)
%% =====================================================================
function mu = meanOf(v)
mu = mySum(v(:)) / numel(v);
end

function [m, idx] = maxOf(v)
m = v(1);  idx = 1;
for k = 2:numel(v)
    if v(k) > m, m = v(k); idx = k; end
end
end

function H = entropyOf(p)
p = p(p > 0);  H = -mySum(p .* myLog(p));
end

function s = mySort(x)
% Bottom-up merge sort, ascending.
x = x(:)';  n = numel(x);  w = 1;
while w < n
    out = x;
    for lo = 1:2 * w:n
        mid = myMin(lo + w - 1, n);  hi = myMin(lo + 2 * w - 1, n);
        i = lo;  j = mid + 1;  k = lo;
        while i <= mid && j <= hi
            if x(i) <= x(j), out(k) = x(i); i = i + 1; else, out(k) = x(j); j = j + 1; end
            k = k + 1;
        end
        while i <= mid, out(k) = x(i); i = i + 1; k = k + 1; end
        while j <= hi,  out(k) = x(j); j = j + 1; k = k + 1; end
    end
    x = out;  w = 2 * w;
end
s = x;
end

function q = quantileOf(sorted, p)
% Linear-interpolated quantile of an already sorted vector.
n = numel(sorted);  h = (n - 1) * p + 1;
lo = myFloor(h);  hi = myMin(lo + 1, n);
q = sorted(lo) + (h - lo) * (sorted(hi) - sorted(lo));
end

function out = describe(v)
% [mean median std skewness kurtosis min max], std with n-1
v = v(:);  n = numel(v);  s = mySort(v);
mu = meanOf(v);  d = v - mu;
sd = mySqrt(mySum(d.^2) / (n - 1));
m2 = meanOf(d.^2);
out = [mu, quantileOf(s, 0.5), sd, meanOf(d.^3) / m2^1.5, meanOf(d.^4) / m2^2, s(1), s(end)];
end

function d = cohenD(a, b)
% Effect size: difference of means over the pooled standard deviation
na = numel(a);  nb = numel(b);
va = mySum((a - meanOf(a)).^2) / (na - 1);  vb = mySum((b - meanOf(b)).^2) / (nb - 1);
d = (meanOf(a) - meanOf(b)) / mySqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2));
end

function [t, df, lp, lo, hi] = welchT(a, b)
% Welch two-sample t-test (unequal variances), two-sided.
%   t  = (mean_a - mean_b) / se,   se = sqrt(s_a^2/n_a + s_b^2/n_b)
%   df = se^4 / ((s_a^2/n_a)^2/(n_a-1) + (s_b^2/n_b)^2/(n_b-1))   (Welch-Satterthwaite)
%   95 % CI of the difference = diff +/- t_{0.975,df} * se
na = numel(a);  nb = numel(b);
va = mySum((a - meanOf(a)).^2) / (na - 1);  vb = mySum((b - meanOf(b)).^2) / (nb - 1);
se = mySqrt(va / na + vb / nb);
dm = meanOf(a) - meanOf(b);
t  = dm / se;
df = (va / na + vb / nb)^2 / ((va / na)^2 / (na - 1) + (vb / nb)^2 / (nb - 1));
lp = log10TwoSidedT(t, df);
tc = tCritical(0.05, df);
lo = dm - tc * se;  hi = dm + tc * se;
end

function lp = log10TwoSidedT(t, df)
% Two-sided p-value of Student's t as log10:  p = I_x(df/2, 1/2),  x = df/(df + t^2),
% where I is the regularised incomplete beta function. Logs avoid underflow for huge |t|.
x = df / (df + t^2);  a = df / 2;  b = 0.5;
if x < (a + 1) / (a + b + 2)
    lI = logGamma(a + b) - logGamma(a) - logGamma(b) + a * myLog(x) + b * myLog(1 - x) + ...
         myLog(betaCF(x, a, b) / a);
else
    bt = myExp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * myLog(x) + b * myLog(1 - x));
    lI = myLog(myMax(1 - bt * betaCF(1 - x, b, a) / b, 1e-300));
end
lp = lI / myLog(10);
end

function h = betaCF(x, a, b)
% Continued fraction for the incomplete beta function (modified Lentz method).
tiny = 1e-300;  qab = a + b;  qap = a + 1;  qam = a - 1;
c = 1;  d = 1 - qab * x / qap;
if myAbs(d) < tiny, d = tiny; end
d = 1 / d;  h = d;
for m = 1:500
    m2 = 2 * m;
    aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d;  if myAbs(d) < tiny, d = tiny; end
    c = 1 + aa / c;  if myAbs(c) < tiny, c = tiny; end
    d = 1 / d;  h = h * d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d;  if myAbs(d) < tiny, d = tiny; end
    c = 1 + aa / c;  if myAbs(c) < tiny, c = tiny; end
    d = 1 / d;  del = d * c;  h = h * del;
    if myAbs(del - 1) < 1e-15, break; end
end
end

function g = logGamma(x)
% log Gamma(x), x > 0, by the Lanczos approximation (g = 5, six terms; Press et al., 2007),
% relative error below 2e-10.
c = [76.18009172947146, -86.50532032941677, 24.01409824083091, ...
     -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
tmp = x + 5.5;  tmp = tmp - (x + 0.5) * myLog(tmp);
ser = 1.000000000190015;
for j = 1:6, ser = ser + c(j) / (x + j); end
g = -tmp + myLog(2.5066282746310005 * ser / x);
end

function le = logErfc(x)
% log erfc(x) for x >= 0 by the Chebyshev-fitted formula of Press et al. (2007),
% erfc(x) = u exp(-x^2 - 1.26551223 + u(1.00002368 + ...)),  u = 1/(1 + x/2),
% relative error below 1.2e-7. Working in logs keeps tiny tails (p ~ 1e-100) from underflowing.
u = 1 / (1 + 0.5 * x);
c = [-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, ...
     0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277];
pol = c(10);
for j = 9:-1:1, pol = c(j) + u * pol; end
le = myLog(u) - x ^ 2 + pol;
end

function tc = tCritical(alpha, df)
% t value whose two-sided p equals alpha, found by bisection on the p-value.
lo = 0;  hi = 100;
for it = 1:200
    mid = (lo + hi) / 2;
    if 10^log10TwoSidedT(mid, df) > alpha, lo = mid; else, hi = mid; end
end
tc = (lo + hi) / 2;
end

function [U, z, lp, rb] = mannWhitney(a, b)
% Mann-Whitney U test (two-sided), normal approximation with tie correction.
%   U = R_a - n_a(n_a+1)/2,  z = (U - n_a n_b/2) / sigma_U,
%   sigma_U^2 = n_a n_b/12 * ((n+1) - sum(t^3 - t)/(n(n-1)))
%   rank-biserial r = 2U/(n_a n_b) - 1  (effect size, -1..1)
na = numel(a);  nb = numel(b);  n = na + nb;
[s, idx] = mySortIdx([a(:); b(:)]);
r = zeros(1, n);  tieSum = 0;  i = 1;
while i <= n                                  % average ranks over runs of ties
    j = i;
    while j < n && s(j + 1) == s(i), j = j + 1; end
    r(idx(i:j)) = (i + j) / 2;
    tl = j - i + 1;  tieSum = tieSum + tl^3 - tl;
    i = j + 1;
end
U  = mySum(r(1:na)) - na * (na + 1) / 2;
sg = mySqrt(na * nb / 12 * ((n + 1) - tieSum / (n * (n - 1))));
z  = (U - na * nb / 2) / sg;
lp = logErfc(myAbs(z) / mySqrt(2)) / myLog(10);    % two-sided normal tail, p = erfc(|z|/sqrt 2)
rb = 2 * U / (na * nb) - 1;
end

function [x, idx] = mySortIdx(x)
% Bottom-up merge sort (ascending) that also returns the original positions.
x = x(:)';  n = numel(x);  idx = 1:n;  w = 1;
while w < n
    ox = x;  oi = idx;
    for lo = 1:2 * w:n
        mid = myMin(lo + w - 1, n);  hi = myMin(lo + 2 * w - 1, n);
        i = lo;  j = mid + 1;  k = lo;
        while i <= mid && j <= hi
            if x(i) <= x(j), ox(k) = x(i); oi(k) = idx(i); i = i + 1;
            else, ox(k) = x(j); oi(k) = idx(j); j = j + 1; end
            k = k + 1;
        end
        while i <= mid, ox(k) = x(i); oi(k) = idx(i); i = i + 1; k = k + 1; end
        while j <= hi,  ox(k) = x(j); oi(k) = idx(j); j = j + 1; k = k + 1; end
    end
    x = ox;  idx = oi;  w = 2 * w;
end
end

function s = pText(lp)
% Format a log10 p-value for a plot label.
if lp < -300, s = '< 1e-300'; else, s = sprintf('= %.1e', 10^lp); end
end

function Z = zscore(F)
% Standardise every column: z = (x - mu) / sigma
Z = zeros(size(F));
for j = 1:size(F, 2)
    v = F(:, j);  mu = meanOf(v);  sd = mySqrt(mySum((v - mu).^2) / (numel(v) - 1));
    Z(:, j) = (v - mu) / myMax(sd, 1e-12);
end
end

function [V, lam] = jacobiEig(A)
% Cyclic Jacobi rotations diagonalise a symmetric matrix: A = V diag(lam) V'
n = size(A, 1);  V = myEye(n);
for sweep = 1:100
    off = 0;
    for p = 1:n - 1
        for q = p + 1:n
            off = off + A(p, q)^2;
        end
    end
    if off < 1e-22, break; end
    for p = 1:n - 1
        for q = p + 1:n
            if myAbs(A(p, q)) < 1e-15, continue; end
            theta = (A(q, q) - A(p, p)) / (2 * A(p, q));
            t = 1 / (myAbs(theta) + mySqrt(theta^2 + 1));
            if theta < 0, t = -t; end
            c = 1 / mySqrt(t^2 + 1);  s = t * c;
            Jr = myEye(n);  Jr(p, p) = c;  Jr(q, q) = c;  Jr(p, q) = s;  Jr(q, p) = -s;
            A = Jr' * A * Jr;  V = V * Jr;
        end
    end
end
lam = zeros(n, 1);
for k = 1:n, lam(k) = A(k, k); end
end

function order = descendingOrder(v)
% Indices that sort v from largest to smallest (selection sort).
n = numel(v);  order = 1:n;  v = v(:)';
for i = 1:n - 1
    [~, k] = maxOf(v(i:n));  k = k + i - 1;
    t = v(i); v(i) = v(k); v(k) = t;
    t = order(i); order(i) = order(k); order(k) = t;
end
end

function [x, d] = kde(v, nGrid)
% Gaussian kernel density estimate, Silverman bandwidth h = 1.06 sigma n^(-1/5)
v = v(:);  n = numel(v);
sd = mySqrt(mySum((v - meanOf(v)).^2) / (n - 1));
h = myMax(1.06 * sd * n^(-1 / 5), 1e-9);
s = mySort(v);
x = myLinspace(s(1), s(end), nGrid);
d = zeros(1, nGrid);
for k = 1:nGrid
    d(k) = mySum(myExp(-0.5 * ((x(k) - v) / h).^2)) / (n * h * mySqrt(2 * PI()));
end
end

%% =====================================================================
%%                     TASK 5: figures
%% =====================================================================
function saveFig(fig, folder, name)
print(fig, fullfile(folder, [name '.png']), '-dpng', '-r150');
close(fig);
end

function plotDataset(dataDir, out, cand, sel, y, COL6, NAMES6, pixCount, pureCount, P)
% Fig 1: example image, its mask, and the patches taken from it.
% Fig 2: class pixel share and pure-patch availability.
imgId = mode1(cand(sel, 1));
I = imread(fullfile(dataDir, 'images', sprintf('image%04d.jpg', imgId)));
M = imread(fullfile(dataDir, 'masks',  sprintf('mask%04d.png',  imgId)));
fig = figure('visible', 'off', 'position', [50 50 1400 620]);
subplot(1, 2, 1); image(I); axis image off; hold on;
title(sprintf('Image %04d with sampled patches (cyan = road, purple = building)', imgId));
for j = 1:numel(sel)
    if cand(sel(j), 1) ~= imgId, continue; end
    if y(j) == 1, ec = COL6(3, :) / 255; else, ec = [0.75 0.35 1]; end
    rectangle('position', [cand(sel(j), 3), cand(sel(j), 2), P, P], 'edgecolor', ec, 'linewidth', 1.5);
end
subplot(1, 2, 2); image(M); axis image off; title('Ground-truth mask');
hold on;
for c = 1:6
    plot(nan, nan, 's', 'markersize', 12, 'markerfacecolor', COL6(c, :) / 255, ...
         'markeredgecolor', 'k');
end
legend(NAMES6, 'location', 'eastoutside');
saveFig(fig, out, 'fig1_example_image_mask');

fig = figure('visible', 'off', 'position', [50 50 1300 480]);
subplot(1, 2, 1);
share = 100 * pixCount / mySum(pixCount);
colourBars(share, COL6 / 255);
set(gca, 'xticklabel', NAMES6);  ylabel('Share of labelled pixels (%)');
title('Pixel share per class (202 valid masks)');
for c = 1:6, text(c, share(c) + 1, sprintf('%.1f%%', share(c)), 'horizontalalignment', 'center'); end
subplot(1, 2, 2);
colourBars(myLog10(myMax(pureCount, 1)), COL6 / 255);  ylim([0 6]);
set(gca, 'xticklabel', NAMES6);  ylabel('log_{10}(number of pure 64x64 patches)');
title('Pure patches available (\geq 60% one class)');
for c = 1:6, text(c, myLog10(myMax(pureCount(c), 1)) + 0.12, sprintf('%d', pureCount(c)), 'horizontalalignment', 'center'); end
saveFig(fig, out, 'fig2_class_balance');
end

function colourBars(v, colours)
% one filled rectangle per class, each in its mask colour
hold on;
for c = 1:numel(v)
    fill([c - 0.35, c + 0.35, c + 0.35, c - 0.35], [0 0 v(c) v(c)], colours(c, :), 'edgecolor', 'k');
end
set(gca, 'xtick', 1:numel(v));  xlim([0.4 numel(v) + 0.6]);
end

function u = uniqueInts(v)
% sorted distinct values of a vector of positive integers (counting method)
seen = false(1, myMax(v));  seen(v) = true;  u = myFind(seen);
end

function m = mode1(v)
% most frequent value of a vector of positive integers
cnt = zeros(1, myMax(v));
for k = 1:numel(v), cnt(v(k)) = cnt(v(k)) + 1; end
[~, m] = maxOf(cnt);
end

function plotSamples(X, y, out)
% Fig 3: 10 random road and 10 random building patches.
ir = myFind(y == 1);  ib = myFind(y == 0);
fig = figure('visible', 'off', 'position', [50 50 1300 300]);
for k = 1:10
    subplot(2, 10, k);      image(X(:, :, :, ir(k * 7)));  axis image off;
    if k == 1, title('Road'); end
    subplot(2, 10, 10 + k); image(X(:, :, :, ib(k * 7)));  axis image off;
    if k == 1, title('Building'); end
end
saveFig(fig, out, 'fig3_sample_patches');
end

function plotHistograms(X, y, out, cr, cb)
% Fig 4: pixel-intensity distributions per class for R, G, B and luminance.
ch = {'Red', 'Green', 'Blue', 'Luminance Y'};
fig = figure('visible', 'off', 'position', [50 50 1300 330]);
D = double(X) / 255;
Yl = 0.299 * D(:, :, 1, :) + 0.587 * D(:, :, 2, :) + 0.114 * D(:, :, 3, :);
for b = 1:4
    subplot(1, 4, b); hold on;
    for k = [1 0]
        if b < 4, v = D(:, :, b, y == k); else, v = Yl(:, :, 1, y == k); end
        v = myFloor(v(:) * 31.999);
        h = zeros(1, 32);
        for g = 0:31, h(g + 1) = mySum(v == g); end
        h = h / mySum(h);
        xs = ((0:31) + 0.5) / 32;
        if k == 1, c = cr; else, c = cb; end
        fill([xs(1) xs xs(end)], [0 h 0], c, 'facealpha', 0.45, 'edgecolor', c);
    end
    title(ch{b}); xlabel('Intensity (0-1)'); ylabel('Probability'); xlim([0 1]);
    if b == 1, legend({'Road', 'Building'}); end
end
saveFig(fig, out, 'fig4_intensity_histograms');
end

function plotPipeline(X, y, out)
% Fig 5: the feature-engineering pipeline on one road and one building patch.
ir = myFind(y == 1);  ib = myFind(y == 0);
ids = [ir(3), ib(3)];
lab = {'Road', 'Building'};
fig = figure('visible', 'off', 'position', [50 50 1900 440]);
cols = {'RGB patch', 'Median + Gaussian Y', 'Sobel magnitude', '|Laplacian|', 'Otsu binary', ...
        'Opened (3x3)', 'Black top-hat', 'log |DFT| (centred)', 'GLCM (8 levels)', 'HOG 2x2 cells x 9 bins'};
for r = 1:2
    I = double(X(:, :, :, ids(r))) / 255;
    Y = 0.299 * I(:, :, 1) + 0.587 * I(:, :, 2) + 0.114 * I(:, :, 3);
    Ys = preprocess(Y);
    TH = greyErode3(greyErode3(greyDilate3(greyDilate3(Ys)))) - Ys;
    Gx = conv2same(Ys, [-1 0 1; -2 0 2; -1 0 1]);  Gy = conv2same(Ys, [-1 -2 -1; 0 0 0; 1 2 1]);
    B = Ys > otsuThreshold(Ys);
    N = size(Ys, 1);
    F = mySqrt(dftPower(Ys));
    F = F([N / 2 + 1:N, 1:N / 2], [N / 2 + 1:N, 1:N / 2]);
    q = myMin(myFloor(Ys * 8), 7) + 1;  G = zeros(8);
    a = q(:, 1:end - 1);  b = q(:, 2:end);
    for k = 1:numel(a), G(a(k), b(k)) = G(a(k), b(k)) + 1; end
    Lp = myAbs(conv2same(Ys, [0 1 0; 1 -4 1; 0 1 0]));
    Hg = hogCells(mySqrt(Gx.^2 + Gy.^2), myAtan2(Gy, Gx), 2, 9);          % rows = cells, cols = bins
    panels = {X(:, :, :, ids(r)), Ys, mySqrt(Gx.^2 + Gy.^2), Lp, B, openBinary(B), TH, myLog(1 + F), G + G', Hg};
    for c = 1:10
        subplot(2, 10, (r - 1) * 10 + c);
        if c == 1, image(panels{c}); else, imagesc(panels{c}); end
        if c == 10, axis normal off; else, axis image off; end
        colormap(gray);
        if r == 1, title(cols{c}, 'fontsize', 9); end
        if c == 1, text(-20, 32, lab{r}, 'rotation', 90, 'horizontalalignment', 'center', 'fontweight', 'bold'); end
    end
end
saveFig(fig, out, 'fig5_feature_pipeline');
end

function plotViolins(FEAT, y, names, D, LPT, LPU, out, cr, cb)
% Fig 6: violin + box + median + mean for the 8 features with the largest |Cohen's d|,
% titled with the effect size and the Welch t and Mann-Whitney p-values.
order = descendingOrder(myAbs(D));
top = order(1:8);
fig = figure('visible', 'off', 'position', [50 50 1400 640]);
for s = 1:8
    f = top(s);
    subplot(2, 4, s); hold on;
    for k = 1:2
        v = FEAT(y == (k == 1), f);
        if k == 1, c = cr; else, c = cb; end
        [xg, d] = kde(v, 80);
        w = 0.38 * d / myMax(d);
        fill([k - w, k + w(end:-1:1)], [xg, xg(end:-1:1)], c, 'facealpha', 0.45, 'edgecolor', c);
        sv = mySort(v);
        q1 = quantileOf(sv, 0.25);  q2 = quantileOf(sv, 0.5);  q3 = quantileOf(sv, 0.75);
        patch([k - 0.06, k + 0.06, k + 0.06, k - 0.06], [q1 q1 q3 q3], 'w', 'edgecolor', 'k');
        plot([k - 0.06, k + 0.06], [q2 q2], 'k', 'linewidth', 2);
        plot(k, meanOf(v), 'kd', 'markerfacecolor', 'y', 'markersize', 5);
    end
    set(gca, 'xtick', [1 2], 'xticklabel', {'Road', 'Building'});  xlim([0.4 2.6]);
    title({sprintf('%s   (d = %.2f)', names{f}, D(f)), ...
           sprintf('Welch p %s,  M-W p %s', pText(LPT(f)), pText(LPU(f)))}, ...
          'interpreter', 'none', 'fontsize', 9);
end
saveFig(fig, out, 'fig6_feature_violins');
end

function plotCorrelation(C, names, out)
% Fig 7: feature correlation heat map (motivates PCA / feature selection).
fig = figure('visible', 'off', 'position', [50 50 900 800]);
imagesc(C, [-1 1]); axis image;
n = 64;  t = myLinspace(0, 1, n)';
colormap([[t; ones(n, 1)], [t; 1 - t], [ones(n, 1); 1 - t]]);   % blue-white-red
set(gca, 'xtick', 1:numel(names), 'ytick', 1:numel(names), 'fontsize', 8);
lbl = cell(1, numel(names));
for k = 1:numel(names), lbl{k} = sprintf('%d %s', k, names{k}); end
set(gca, 'yticklabel', lbl);  xlabel('Feature number');
set(gca, 'position', [0.22 0.1 0.6 0.8]);
colorbar('position', [0.88 0.1 0.03 0.8]);
title(sprintf('Correlation between the %d engineered features', numel(names)));
saveFig(fig, out, 'fig7_feature_correlation');
end

function plotPCA(PC, ex, y, pOut, out, cr, cb)
% Fig 8: scree + 2-D PC scatter.  Fig 9: 3-D PC scatter (overlap in one view).
fig = figure('visible', 'off', 'position', [50 50 1300 500]);
subplot(1, 2, 1);
cum = zeros(size(ex));  acc = 0;
for k = 1:numel(ex), acc = acc + ex(k); cum(k) = acc; end
bar(100 * ex(1:10), 'facecolor', [0.12 0.44 0.55]); hold on;
plot(1:10, 100 * cum(1:10), 'o-', 'color', [0.99 0.64 0.07], 'linewidth', 2);
xlabel('Principal component'); ylabel('Variance explained (%)');
legend({'Each PC', 'Cumulative'}, 'location', 'east'); title('PCA scree plot');
subplot(1, 2, 2); hold on;
scatter(PC(y == 0, 1), PC(y == 0, 2), 14, cb, 'filled', 'markerfacealpha', 0.5);
scatter(PC(y == 1, 1), PC(y == 1, 2), 14, cr, 'filled', 'markerfacealpha', 0.6);
xlabel(sprintf('PC1 (%.1f%%)', 100 * ex(1))); ylabel(sprintf('PC2 (%.1f%%)', 100 * ex(2)));
legend({'Building', 'Road'}); title('Patches in PC1-PC2 space');
saveFig(fig, out, 'fig8_pca_2d');

% Fig 9 encodes five variables in one 3-D view: PC1-PC3 (position), class (colour),
% model confidence |2 P(road) - 1| (marker size); rings mark the class centroids.
fig = figure('visible', 'off', 'position', [50 50 900 750]); hold on;
conf = myAbs(2 * pOut - 1);
for cls = [0 1]
    m = y == cls;
    if cls == 1, c = cr; else, c = cb; end
    scatter3(PC(m, 1), PC(m, 2), PC(m, 3), 4 + 34 * conf(m), c, 'filled', 'markerfacealpha', 0.35);
end
lbl = {'building centroid', 'road centroid'};
for cls = [0 1]
    m = y == cls;
    if cls == 1, c = cr; else, c = cb; end
    ctr = [meanOf(PC(m, 1)), meanOf(PC(m, 2)), meanOf(PC(m, 3))];
    scatter3(ctr(1), ctr(2), ctr(3), 420, c, 'filled', 'markeredgecolor', 'k', 'linewidth', 2.5);
    if cls == 1, ha = 'right'; else, ha = 'left'; end         % road label left, building right
    text(ctr(1), ctr(2), ctr(3) + 3.5, lbl{cls + 1}, 'fontweight', 'bold', 'fontsize', 11, ...
         'horizontalalignment', ha, 'backgroundcolor', 'w', 'edgecolor', c);
end
xlabel(sprintf('PC1 (%.1f%%)', 100 * ex(1))); ylabel(sprintf('PC2 (%.1f%%)', 100 * ex(2)));
zlabel(sprintf('PC3 (%.1f%%)', 100 * ex(3)));
legend({'Building', 'Road'}, 'location', 'northeast');
grid on; view(25, 24);                                  % looks across the PC1-PC2 class axis
title({'3-D feature space: first three principal components', ...
       'marker size = model confidence |2P(road) - 1| (out-of-fold)'});
saveFig(fig, out, 'fig9_pca_3d');
end

function plotViolinsOneAxis(FEAT, y, names, D, LPT, LPU, out, cr, cb)
% Fig 11: the 8 features with the largest |Cohen's d| on ONE axis. Each feature is
% rescaled by its own z-score, z = (x - mu)/sigma, so different units share a scale.
% Split violins (road left, building right), IQR box, median bar, mean diamond,
% and the effect size and the larger of the two test p-values above each pair.
Z = zscore(FEAT);
order = descendingOrder(myAbs(D));  top = order(1:8);
fig = figure('visible', 'off', 'position', [50 50 1500 620]); hold on;
plot([0.4 8.6], [0 0], ':', 'color', [0.6 0.6 0.6]);
for s = 1:8
    f = top(s);
    for k = 1:2                                   % k = 1 road (left), 2 building (right)
        v = Z(y == (k == 1), f);
        v = v(myAbs(v) <= 4);                       % clip extreme tails for the plot only
        if k == 1, c = cr; side = -1; else, c = cb; side = 1; end
        [xg, d] = kde(v, 80);
        w = 0.42 * d / maxOf(d);
        fill([s + 0 * w, s + side * w(end:-1:1)], [xg, xg(end:-1:1)], c, ...
             'facealpha', 0.7, 'edgecolor', c);
        sv = mySort(v);
        q1 = quantileOf(sv, 0.25);  q2 = quantileOf(sv, 0.5);  q3 = quantileOf(sv, 0.75);
        xc = s + side * 0.1;
        patch([xc - 0.04, xc + 0.04, xc + 0.04, xc - 0.04], [q1 q1 q3 q3], 'w', 'edgecolor', 'k');
        plot([xc - 0.04, xc + 0.04], [q2 q2], 'k', 'linewidth', 2);
        plot(xc, meanOf(v), 'kd', 'markerfacecolor', 'y', 'markersize', 6);
    end
    lp = LPT(f);  if LPU(f) > lp, lp = LPU(f); end          % report the weaker test
    text(s, 4.55, sprintf('d = %+.2f', D(f)), 'horizontalalignment', 'center', ...
         'fontweight', 'bold', 'fontsize', 10);
    text(s, 4.15, sprintf('p < 1e%d', -myFloor(-lp)), 'horizontalalignment', 'center', 'fontsize', 9);
end
set(gca, 'xtick', 1:8, 'xticklabel', names(top), 'fontsize', 10);
xlim([0.4 8.6]);  ylim([-4.2 5]);  ylabel('z-score  (x - \mu) / \sigma');
h1 = fill(nan, nan, cr);  h2 = fill(nan, nan, cb);
legend([h1 h2], {'Road (left half)', 'Building (right half)'}, 'location', 'southeast');
title('Top 8 features on one axis: box = IQR, bar = median, diamond = mean, d = Cohen''s d');
saveFig(fig, out, 'fig11_violins_one_axis');
end

function [fpr, tpr, auc] = rocCurve(p, y)
% ROC curve: sort scores from high to low and step the threshold down,
% TPR = TP/P, FPR = FP/N; AUC by the trapezoid rule. Tied scores share one
% threshold, so a group of ties is one diagonal step (k-NN, naive Bayes and
% tree scores have many ties; stepping through them one by one inflates the AUC).
[ps, idx] = mySortIdx(-p(:)');
yy = reshape(y(idx) == 1, 1, []);            % row vector of labels, best score first
P = mySum(yy);  Nn = numel(yy) - P;
cut = [ps(1:end - 1) ~= ps(2:end), true];   % last position of each tied group
tp = myCumsum(double(yy));  fp = myCumsum(double(~yy));
tpr = [0, tp(cut) / P];
fpr = [0, fp(cut) / Nn];
auc = mySum((fpr(2:end) - fpr(1:end - 1)) .* (tpr(2:end) + tpr(1:end - 1)) / 2);
end

function [lo, hi] = aucCI(A, n1, n0)
% Hanley & McNeil (1982) standard error of the AUC, 95 % normal interval.
q1 = A / (2 - A);  q2 = 2 * A^2 / (1 + A);
se = mySqrt((A * (1 - A) + (n1 - 1) * (q1 - A^2) + (n0 - 1) * (q2 - A^2)) / (n1 * n0));
lo = A - 1.96 * se;  hi = A + 1.96 * se;
if hi > 1, hi = 1; end
end

function [lo, hi] = wilsonCI(k, n)
% Wilson score 95 % interval for a proportion k/n (z = 1.96).
z = 1.96;  ph = k / n;
c = (ph + z^2 / (2 * n)) / (1 + z^2 / n);
h = z * mySqrt(ph * (1 - ph) / n + z^2 / (4 * n^2)) / (1 + z^2 / n);
lo = c - h;  hi = c + h;
end

function lp = log10BinomTail(k, n)
% Exact one-sided binomial test of H0: success rate 0.5,  p = P(X >= k), X ~ Bin(n, 0.5).
% log C(n, i) from cumulative sums of logs, then log-sum-exp to avoid underflow.
lf = [0, myCumsum(myLog(1:n))];                       % lf(i + 1) = log(i!)
lt = zeros(1, n - k + 1);
for i = k:n
    lt(i - k + 1) = lf(n + 1) - lf(i + 1) - lf(n - i + 1) - n * myLog(2);
end
m = maxOf(lt);
lp = (m + myLog(mySum(myExp(lt - m)))) / myLog(10);
end

function plotROC(fpr, tpr, auc, lo, hi, spec, rec, out)
% Fig 12: out-of-fold ROC curve of the white-box model, chance line and the
% operating point used for the confusion matrix (threshold 0.5).
fig = figure('visible', 'off', 'position', [50 50 620 560]); hold on;
fill([fpr, 1], [tpr, 0], [0.43 0.76 0.89], 'facealpha', 0.25, 'edgecolor', 'none');
plot(fpr, tpr, '-', 'color', [0.12 0.44 0.55], 'linewidth', 2.5);
plot([0 1], [0 1], 'k--');
plot(1 - spec, rec, 'o', 'markersize', 10, 'markerfacecolor', [0.99 0.64 0.07], 'markeredgecolor', 'k');
text(1 - spec + 0.04, rec - 0.05, 'threshold 0.5', 'fontsize', 10);
axis square;  xlim([0 1]);  ylim([0 1]);  grid on;
xlabel('False-positive rate (1 - specificity)');  ylabel('True-positive rate (recall)');
title(sprintf('ROC, grouped 5-fold CV: AUC = %.3f (95%% CI %.3f-%.3f)', auc, lo, hi));
saveFig(fig, out, 'fig12_roc');
end

%% =====================================================================
%%            XAI preview: white-box model and explanations
%% =====================================================================
function mdl = trainLogReg(F, y, lambda)
% Logistic regression by batch gradient descent on standardised features.
%   loss = mean( -y log p - (1-y) log(1-p) ) + lambda/2 |w|^2
%   grad_w = Z'(p - y)/n + lambda w,   grad_b = mean(p - y)
n = size(F, 1);  nF = size(F, 2);
mu = zeros(1, nF);  sd = zeros(1, nF);
for j = 1:nF
    mu(j) = meanOf(F(:, j));  sd(j) = myMax(mySqrt(mySum((F(:, j) - mu(j)).^2) / (n - 1)), 1e-12);
end
Z = (F - mu) ./ sd;
w = zeros(nF, 1);  b = 0;  lr = 0.5;
for it = 1:3000
    p = 1 ./ (1 + myExp(-(Z * w + b)));
    w = w - lr * (Z' * (p - y) / n + lambda * w);
    b = b - lr * meanOf(p - y);
end
mdl = struct('w', w, 'b', b, 'mu', mu, 'sd', sd);
end

function p = predictLogReg(mdl, F)
p = 1 ./ (1 + myExp(-(((F - mdl.mu) ./ mdl.sd) * mdl.w + mdl.b)));
end

function plotXAI(X, FEAT, names, mdl, PIm, CF, acc, out, cr, cb)
% Fig 10: (a) permutation importance (mean and fold range), (b) the model's own
% standardised weights, (c, d) counterfactual explanations for two patches.
fig = figure('visible', 'off', 'position', [50 50 1500 560]);
K = size(PIm, 1);
pm = mySum(PIm, 1) / K;  order = descendingOrder(pm);  top = order(1:10);
axes('position', [0.09 0.11 0.17 0.76]); hold on;
for r = 1:10
    f = top(r);  yy = 11 - r;
    fill([0 pm(f) pm(f) 0], yy + [-0.35 -0.35 0.35 0.35], [0.12 0.44 0.55], 'edgecolor', 'none');
    plot([-maxOf(-PIm(:, f)') maxOf(PIm(:, f)')], [yy yy], 'k-', 'linewidth', 1.2);
end
set(gca, 'ytick', 1:10, 'yticklabel', names(top(10:-1:1)), 'fontsize', 8);  ylim([0.4 10.6]);
xlabel('Accuracy lost when shuffled');
title(sprintf('(a) Permutation importance\n(grouped CV, accuracy %.2f)', acc), 'fontsize', 9);
order = descendingOrder(myAbs(mdl.w'));  top = order(1:10);
axes('position', [0.36 0.11 0.17 0.76]); hold on;
for r = 1:10
    f = top(r);  yy = 11 - r;
    if mdl.w(f) > 0, c = cr; else, c = cb; end
    fill([0 mdl.w(f) mdl.w(f) 0], yy + [-0.35 -0.35 0.35 0.35], c, 'edgecolor', 'k');
end
plot([0 0], [0.4 10.6], 'k-');
set(gca, 'ytick', 1:10, 'yticklabel', names(top(10:-1:1)), 'fontsize', 8);  ylim([0.4 10.6]);
xlabel('Weight per SD (+ road, - building)');
title(sprintf('(b) White-box weights\n(logistic regression)'), 'fontsize', 9);
lab = {'Road', 'Building'};
for k = 1:2
    j = CF(k, 1);  f = CF(k, 2);
    axes('position', [0.56 + 0.22 * k - 0.22, 0.11, 0.2, 0.6]);  image(X(:, :, :, j));  axis image off;
    if k == 1, other = 'building'; else, other = 'road'; end
    title(sprintf('(%s) %s patch, P(road) = %.2f\nwould be called %s if\n%s went %.3g -> %.3g', ...
          char('b' + k), lab{k}, CF(k, 4), other, names{f}, FEAT(j, f), FEAT(j, f) + CF(k, 3)), ...
          'interpreter', 'none', 'fontsize', 9);
end
saveFig(fig, out, 'fig10_xai_explanations');
end

%% =====================================================================
%%            TASK 6: model comparison (hand-written classifiers)
%% =====================================================================
function best = tuneInner(kind, Z, y, fold, grid)
% Nested CV: each training fold is held out once, every grid value is scored on
% it, and the value with the highest mean inner accuracy wins (ties -> first value).
fs = uniqueInts(fold);
accs = zeros(numel(fs), numel(grid));
for a = 1:numel(fs)
    tr = fold ~= fs(a);  te = ~tr;
    if strcmp(kind, 'knn')
        S = knnScores(Z(tr, :), y(tr), Z(te, :), grid);
    else                                          % one deep tree, cut at each depth
        T = growTree(Z(tr, :), y(tr), myMax(grid), 5);
        S = zeros(mySum(te), numel(grid));
        for g = 1:numel(grid)
            S(:, g) = predictTree(T, Z(te, :), grid(g));
        end
    end
    accs(a, :) = mySum((S >= 0.5) == (y(te) == 1), 1) / mySum(te);
end
[~, i] = maxOf(mySum(accs, 1));             % first maximum wins a tie
best = grid(i);
end

function S = knnScores(Ztr, ytr, Zte, ks)
% k-NN: score = share of road patches among the k nearest training patches,
%   d(x, x_i)^2 = |x|^2 + |x_i|^2 - 2 x'x_i  (Euclidean, standardised features).
% The neighbour order does not depend on k, so all k in ks are scored at once.
D = mySum(Zte .^ 2, 2) + mySum(Ztr .^ 2, 2)' - 2 * Zte * Ztr';
% Partial selection sort: take the nearest remaining patch max(ks) times,
% for every test row at once, and mark it as used.
K = myMax(ks);  nT = size(D, 1);  Yn = zeros(nT, K);
for k = 1:K
    [~, j] = myMin(D, [], 2);
    Yn(:, k) = ytr(j);
    D((j - 1) * nT + (1:nT)') = inf;
end
C = myCumsum(Yn, 2);
S = C(:, ks) ./ ks;
end

function s = nbScore(Ztr, ytr, Zte)
% Gaussian naive Bayes:
%   log P(road|x) - log P(bld|x) = log(pi_1/pi_0) + sum_j [log N(x_j; mu_1j, v_1j) - log N(x_j; mu_0j, v_0j)]
% returned as the posterior P(road|x).
pr = meanOf(ytr);
L = myLog(pr) - myLog(1 - pr);
for c = [1 0]
    Zc = Ztr(ytr == c, :);
    nc = size(Zc, 1);  mu = mySum(Zc, 1) / nc;
    v = mySum((Zc - mu) .^ 2, 1) / (nc - 1) + 1e-6;   % sample variance; floor keeps v > 0
    ll = -0.5 * mySum(myLog(2 * PI() * v) + (Zte - mu) .^ 2 ./ v, 2);
    if c == 1, L = L + ll; else, L = L - ll; end
end
s = 1 ./ (1 + myExp(-L));
end

function s = ldaScore(Ztr, ytr, Zte)
% Fisher LDA with equal class covariance Sw:
%   score(x) = w'x - (mu_1 + mu_0)'w / 2 + log(pi_1/pi_0),  w = Sw^-1 (mu_1 - mu_0),
% road if score > 0. The colour means correlate at r ~ 1, so Sw is nearly singular;
% it is shrunk towards its diagonal: Sw = (1 - g) Sw + g diag(Sw), g = 0.1.
G = 0.1;
m1 = mySum(Ztr(ytr == 1, :), 1) / mySum(ytr == 1);  m0 = mySum(Ztr(ytr == 0, :), 1) / mySum(ytr == 0);
Rr = [Ztr(ytr == 1, :) - m1; Ztr(ytr == 0, :) - m0];
Sw = (Rr' * Rr) / (size(Ztr, 1) - 2);
Sw = (1 - G) * Sw + G * keepDiagonal(Sw);
w = mySolve(Sw, (m1 - m0)');
pr = meanOf(ytr);
s = Zte * w - 0.5 * (m1 + m0) * w + myLog(pr / (1 - pr));
end

function s = svmScore(Ztr, ytr, Zte)
% Linear SVM in the primal:  min  lambda/2 |w|^2 + mean( max(0, 1 - t_i (w'z_i + b)) ),
% t_i = +1 road / -1 building. Full-batch sub-gradient descent with step 0.5/sqrt(it);
% the average of the second half of the iterates is kept (it is more stable than the last).
LAM = 0.01;  t = 2 * ytr - 1;  n = size(Ztr, 1);
w = zeros(size(Ztr, 2), 1);  b = 0;  wa = w;  ba = 0;  na = 0;
for it = 1:2000
    act = t .* (Ztr * w + b) < 1;                 % points inside the margin
    gw = LAM * w - Ztr(act, :)' * t(act) / n;
    gb = -mySum(t(act)) / n;
    eta = 0.5 / mySqrt(it);
    w = w - eta * gw;  b = b - eta * gb;
    if it > 1000, wa = wa + w;  ba = ba + b;  na = na + 1; end
end
s = Zte * (wa / na) + ba / na;                    % road if s > 0
end

function T = growTree(Z, y, maxDepth, minLeaf)
% CART: every node takes the split (feature j, threshold t) that gives the lowest
% size-weighted Gini impurity  n_L G_L + n_R G_R,  G = 1 - p^2 - (1 - p)^2.
% Nodes are grown breadth-first; each stores its share of road patches.
% Each feature is sorted once (hand-written merge sort); a node keeps the
% sorted order by filtering it to its own rows, so no node sorts again.
n = numel(y);  O = zeros(n, size(Z, 2));
for j = 1:size(Z, 2)
    [~, O(:, j)] = mySortIdx(Z(:, j));
end
T = struct('feat', 0, 'thr', 0, 'left', 0, 'right', 0, 'frac', meanOf(y), 'depth', 0);
rows = {(1:n)'};
todo = 1;
while ~isempty(todo)
    nd = todo(1);  todo(1) = [];
    r = rows{nd};  yr = y(r);
    if T.depth(nd) >= maxDepth || numel(r) < 2 * minLeaf || myAll(yr == yr(1)), continue; end
    inNode = false(n, 1);  inNode(r) = true;
    [j, t] = bestSplit(Z, y, O, inNode, minLeaf);
    if j == 0, continue; end
    goL = Z(r, j) <= t;
    T.feat(nd) = j;  T.thr(nd) = t;
    for side = 1:2
        if side == 1, rr = r(goL); else, rr = r(~goL); end
        m = numel(T.feat) + 1;
        T.feat(m) = 0;  T.thr(m) = 0;  T.left(m) = 0;  T.right(m) = 0;
        T.frac(m) = meanOf(y(rr));  T.depth(m) = T.depth(nd) + 1;
        rows{m} = rr;  todo(end + 1) = m; %#ok<AGROW>
        if side == 1, T.left(nd) = m; else, T.right(nd) = m; end
    end
end
end

function [jBest, tBest] = bestSplit(Z, y, O, inNode, minLeaf)
% Try every feature; in sorted order the left-child counts for each cut are cumulative sums.
n = mySum(inNode);  nPos = mySum(y(inNode));  gBest = inf;  jBest = 0;  tBest = 0;
for j = 1:size(Z, 2)
    o = O(inNode(O(:, j)), j);                   % this node's rows, already sorted
    v = Z(o, j);  cl = myCumsum(y(o));
    i = (minLeaf:n - minLeaf)';
    i = i(v(i) < v(i + 1));                       % cut only between different values
    if isempty(i), continue; end
    pl = cl(i) ./ i;  pr = (nPos - cl(i)) ./ (n - i);
    g = i .* (1 - pl .^ 2 - (1 - pl) .^ 2) + (n - i) .* (1 - pr .^ 2 - (1 - pr) .^ 2);
    [gm, a] = myMin(g);
    if gm < gBest
        gBest = gm;  jBest = j;  tBest = (v(i(a)) + v(i(a) + 1)) / 2;
    end
end
end

function s = predictTree(T, Z, d)
% Follow the splits down to a leaf, or stop at depth d (so one deep tree can be
% evaluated at every smaller depth); the score is that node's share of road patches.
s = zeros(size(Z, 1), 1);
for i = 1:size(Z, 1)
    nd = 1;
    while T.feat(nd) > 0 && T.depth(nd) < d
        if Z(i, T.feat(nd)) <= T.thr(nd), nd = T.left(nd); else, nd = T.right(nd); end
    end
    s(i) = T.frac(nd);
end
end

function r = binaryMetrics(score, thr, y)
% [accuracy, Wilson CI low, CI high, precision, recall, specificity, F1, AUC]
yh = score >= thr;  yt = y == 1;
TP = mySum(yh & yt);  FP = mySum(yh & ~yt);  TN = mySum(~yh & ~yt);  FN = mySum(~yh & yt);
prec = TP / myMax(TP + FP, 1);  rec = TP / myMax(TP + FN, 1);  spec = TN / myMax(TN + FP, 1);
[lo, hi] = wilsonCI(TP + TN, numel(y));
[~, ~, auc] = rocCurve(score, y);
r = [(TP + TN) / numel(y), lo, hi, prec, rec, spec, 2 * prec * rec / myMax(prec + rec, EPS()), auc];
end

function lp = mcnemarExact(b, c)
% Exact McNemar test: under H0 the b + c patches that only one model gets right
% split 50/50, so p = 2 P(X <= min(b, c)), X ~ Bin(b + c, 0.5). Returns log10 p.
n = b + c;
if n == 0, lp = 0; return; end
lf = [0, myCumsum(myLog(1:n))];                     % lf(m + 1) = log(m!)
lt = zeros(1, myMin(b, c) + 1);
for i = 0:myMin(b, c)
    lt(i + 1) = lf(n + 1) - lf(i + 1) - lf(n - i + 1) - n * myLog(2);
end
m = myMax(lt);
lp = myMin(0, myLog10(2) + (m + myLog(mySum(myExp(lt - m)))) / myLog(10));
end

function s = forestScore(Ztr, ytr, Zte, nTree, mtry, seed)
% Random forest (Breiman, 2001): nTree CART trees, each grown to pure leaves on a
% bootstrap sample of the training patches, and each split chosen among only mtry
% randomly picked features. The score is the mean over trees of the leaf's share of
% road patches (with pure leaves this is the share of trees voting "road").
% A bootstrap sample is kept as counts w (how often each row was drawn), so every
% feature is sorted once per fold instead of once per tree.
n = numel(ytr);  O = zeros(n, size(Ztr, 2));
for j = 1:size(Ztr, 2)
    [~, O(:, j)] = mySortIdx(Ztr(:, j));
end
r = seed;  s = zeros(size(Zte, 1), 1);
for b = 1:nTree
    [u, r] = parkMiller(n, r);
    w = accumCounts(myFloor(u * n) + 1, n);        % n draws with replacement
    [T, r] = growRandomTree(Ztr, ytr, O, w, mtry, r);
    s = s + predictTree(T, Zte, inf);
end
s = s / nTree;
end

function [u, r] = parkMiller(k, r)
% k uniform numbers in [0, 1) from the Park-Miller "minimal standard" generator,
%   r_{i+1} = 16807 r_i mod (2^31 - 1)  (every product stays exact in double).
u = zeros(k, 1);
for i = 1:k
    r = myMod(16807 * r, 2147483647);
    u(i) = (r - 1) / 2147483646;
end
end

function w = accumCounts(idx, n)
% w(i) = how many times i appears in idx
w = zeros(n, 1);
for i = 1:numel(idx), w(idx(i)) = w(idx(i)) + 1; end
end

function [T, r] = growRandomTree(Z, y, O, w, mtry, r)
% CART grown until every leaf is pure, on rows weighted by their bootstrap counts w.
% Each node draws mtry of the features without replacement (partial Fisher-Yates).
n = numel(y);  p = size(Z, 2);
T = struct('feat', 0, 'thr', 0, 'left', 0, 'right', 0, ...
           'frac', mySum(w .* y) / mySum(w), 'depth', 0);
rows = {myFind(w > 0)};  todo = 1;
while ~isempty(todo)
    nd = todo(1);  todo(1) = [];
    rr = rows{nd};
    if myAll(y(rr) == y(rr(1))), continue; end       % pure node: leaf
    inNode = false(n, 1);  inNode(rr) = true;
    [u, r] = parkMiller(mtry, r);
    f = 1:p;
    for i = 1:mtry                               % pick mtry features
        k = i + myFloor(u(i) * (p - i + 1));
        t = f(i);  f(i) = f(k);  f(k) = t;
    end
    [j, thr] = bestSplitWeighted(Z, y, O, w, inNode, f(1:mtry));
    if j == 0, continue; end                     % no usable cut: leaf
    goL = Z(rr, j) <= thr;
    T.feat(nd) = j;  T.thr(nd) = thr;
    for side = 1:2
        if side == 1, ch = rr(goL); else, ch = rr(~goL); end
        m = numel(T.feat) + 1;
        T.feat(m) = 0;  T.thr(m) = 0;  T.left(m) = 0;  T.right(m) = 0;
        T.frac(m) = mySum(w(ch) .* y(ch)) / mySum(w(ch));  T.depth(m) = T.depth(nd) + 1;
        rows{m} = ch;  todo(end + 1) = m; %#ok<AGROW>
        if side == 1, T.left(nd) = m; else, T.right(nd) = m; end
    end
end
end

function [jBest, tBest] = bestSplitWeighted(Z, y, O, w, inNode, feats)
% Lowest weighted Gini  W_L G_L + W_R G_R  over the given features; W = summed counts.
W = mySum(w(inNode));  P = mySum(w(inNode) .* y(inNode));
gBest = inf;  jBest = 0;  tBest = 0;
for j = feats
    o = O(inNode(O(:, j)), j);                   % node rows, already sorted by feature j
    v = Z(o, j);  wl = myCumsum(w(o));  cl = myCumsum(w(o) .* y(o));
    i = myFind(v(1:end - 1) < v(2:end));           % cut only between different values
    if isempty(i), continue; end
    nl = wl(i);  nr = W - nl;  pl = cl(i) ./ nl;  pr = (P - cl(i)) ./ nr;
    g = nl .* (1 - pl .^ 2 - (1 - pl) .^ 2) + nr .* (1 - pr .^ 2 - (1 - pr) .^ 2);
    [gm, a] = myMin(g);
    if gm < gBest
        gBest = gm;  jBest = j;  tBest = (v(i(a)) + v(i(a) + 1)) / 2;
    end
end
end

function s = rbfSvmScore(Ztr, ytr, Zte)
% RBF-kernel SVM, f(x) = sum_i beta_i k(x_i, x) + b,  k(x, x') = exp(-|x - x'|^2 / (2 sig^2)).
% Minimises  lambda/2 |f|^2 + mean(max(0, 1 - t_i f(x_i)))  by sub-gradient descent in
% function space (kernel Pegasos, full batch): beta <- (1 - eta lambda) beta + eta/n t_act.
% sig = median distance between training patches (median heuristic); lambda = 1/n (C = 1).
% As in svmScore, the second half of the iterates is averaged.
n = size(Ztr, 1);  t = 2 * ytr - 1;  LAM = 1 / n;
sq = mySum(Ztr .^ 2, 2);
D2 = myMax(sq + sq' - 2 * (Ztr * Ztr'), 0);
sub = 1:4:n;                                     % median over an evenly spaced subset
Ds = mySqrt(D2(sub, sub));  Ds = Ds(upperMask(numel(sub)));
sig = quantileOf(mySort(Ds), 0.5);
K = myExp(-D2 / (2 * sig ^ 2));
beta = zeros(n, 1);  b = 0;  ba = beta;  bb = 0;  na = 0;
for it = 1:3000
    act = t .* (K * beta + b) < 1;
    eta = 1 / (LAM * it);                        % Pegasos step
    beta = (1 - eta * LAM) * beta + eta / n * (t .* act);
    b = b + 0.5 / mySqrt(it) * mySum(t(act)) / n;
    if it > 1500, ba = ba + beta;  bb = bb + b;  na = na + 1; end
end
sq2 = mySum(Zte .^ 2, 2);
Kte = myExp(-myMax(sq2 + sq' - 2 * (Zte * Ztr'), 0) / (2 * sig ^ 2));
s = Kte * (ba / na) + bb / na;                   % road if s > 0
end

function plotModels(names, R, SC, y, out)
% Fig 13: (a) accuracy with its 95 % Wilson CI and the AUC for each model,
% (b) the out-of-fold ROC curves of all eight models on one set of axes.
fig = figure('visible', 'off', 'position', [50 50 1450 540]);
nM = numel(names);
cols = [0.12 0.44 0.55; 0.85 0.37 0.01; 0.46 0.44 0.70; 0.11 0.62 0.47; 0.90 0.67 0.00; 0.65 0.46 0.11; ...
        0.91 0.16 0.54; 0.40 0.40 0.40];
axes('position', [0.17 0.12 0.33 0.76]);  hold on;
for m = 1:nM
    yy = nM + 1 - m;
    fill([0.5 R(m, 1) R(m, 1) 0.5], yy + [-0.3 -0.3 0.3 0.3], cols(m, :), 'edgecolor', 'none');
    plot([R(m, 2) R(m, 3)], [yy yy], 'k-', 'linewidth', 1.5);
    plot(R(m, 8), yy, 'kd', 'markerfacecolor', 'w', 'markersize', 7);
    text(0.51, yy, sprintf('%.3f', R(m, 1)), 'color', 'w', 'fontsize', 8, 'fontweight', 'bold');
end
set(gca, 'ytick', 1:nM, 'yticklabel', names(nM:-1:1), 'fontsize', 9);
xlim([0.5 1]);  ylim([0.4 nM + 0.6]);  box on;
xlabel('Accuracy (bar, line = 95% CI) and AUC (diamond)');
title('(a) Eight classifiers on the same grouped 5-fold CV', 'fontsize', 10);
axes('position', [0.6 0.12 0.36 0.76]);  hold on;
lab = cell(1, nM);
for m = 1:nM
    [fp, tp, a] = rocCurve(SC(:, m), y);
    plot(fp, tp, '-', 'color', cols(m, :), 'linewidth', 1.6);
    lab{m} = sprintf('%s (AUC %.3f)', names{m}, a);
end
plot([0 1], [0 1], 'k:');
legend(lab, 'location', 'southeast', 'fontsize', 8);
axis([0 1 0 1]);  axis square;  box on;
xlabel('False positive rate');  ylabel('True positive rate');
title('(b) Out-of-fold ROC curves', 'fontsize', 10);
saveFig(fig, out, 'fig13_model_comparison');
end

%% =====================================================================
%%   Basic maths, hand-written (no MATLAB maths functions are called)
%% =====================================================================
% Only the language itself is used here: + - * / .^ (integer powers), comparisons,
% indexing, loops, and the array tools size, numel, zeros, ones, reshape, permute.
% The reductions work along one dimension at a time and are vectorised over the
% others, so they stay fast on whole images and feature matrices.

function v = PI()
v = 3.141592653589793;
end

function v = EPS()
v = 2.220446049250313e-16;           % spacing of doubles near 1
end

function d = firstDim(x)
% First dimension longer than 1, as MATLAB's own reductions choose it.
sz = size(x);  d = 1;
while d < numel(sz) && sz(d) == 1, d = d + 1; end
end

function [A, sz, order] = toColumns(x, dim)
% Puts dimension dim first and folds the others into columns.
sz = size(x);
if dim > numel(sz), sz(end + 1:dim) = 1; end
order = [dim, 1:dim - 1, dim + 1:numel(sz)];
cols = 1;
for k = order(2:end), cols = cols * sz(k); end
A = reshape(permute(double(x), order), sz(dim), cols);
end

function y = fromColumns(A, sz, order)
% Undoes toColumns for a result that now has one row.
out = sz(order);  out(1) = 1;
back = zeros(1, numel(order));
back(order) = 1:numel(order);
y = permute(reshape(A, out), back);
end

function s = mySum(x, dim)
% Sum along dim, as a row of ones times the matrix: [1 1 ... 1] * A adds up
% every column of A in one matrix product.
if nargin < 2
    if numel(x) == 0, s = 0; return; end
    if size(x, 1) == 1, dim = 2; else, dim = 1; end
    if numel(size(x)) > 2, dim = firstDim(x); end
end
if numel(size(x)) == 2 && dim == 1
    s = ones(1, size(x, 1)) * double(x);  return
end
if numel(size(x)) == 2 && dim == 2
    s = double(x) * ones(size(x, 2), 1);  return
end
[A, sz, order] = toColumns(x, dim);
s = fromColumns(ones(1, size(A, 1)) * A, sz, order);
end

function c = myCumsum(x, dim)
% Running sum along dim: after the step with shift d every entry holds the sum
% of the 2d entries up to it (log2(n) vectorised steps).
if nargin < 2, dim = firstDim(x); end
if numel(size(x)) == 2 && (dim == 1 || size(x, 1) == 1)
    flip = dim == 2;                     % rows are summed as columns
    if flip, A = double(x)'; else, A = double(x); end
    n = size(A, 1);  d = 1;
    while d < n
        A(d + 1:n, :) = A(d + 1:n, :) + A(1:n - d, :);
        d = 2 * d;
    end
    if flip, c = A'; else, c = A; end
    return
end
[A, sz, order] = toColumns(x, dim);
n = size(A, 1);  d = 1;
while d < n
    A(d + 1:n, :) = A(d + 1:n, :) + A(1:n - d, :);
    d = 2 * d;
end
back = zeros(1, numel(order));
back(order) = 1:numel(order);
c = permute(reshape(A, sz(order)), back);
end

function [m, idx] = myMax(a, b, dim)
% myMax(x): largest value (and its first position) along the first long dimension.
% myMax(a, b): the larger of a and b, element by element.  myMax(x, [], dim).
if nargin == 2, m = pickLarger(a, b, true); return; end
if nargin < 3, dim = firstDim(a); end
[m, idx] = extremeAlong(a, dim, true);
end

function [m, idx] = myMin(a, b, dim)
% Same as myMax, for the smallest value.
if nargin == 2, m = pickLarger(a, b, false); return; end
if nargin < 3, dim = firstDim(a); end
[m, idx] = extremeAlong(a, dim, false);
end

function m = pickLarger(a, b, wantMax)
% Element-wise max or min of a and b (scalars expand); a NaN loses to a number.
% Adding zeros of the other's size expands a scalar without touching Inf.
m = double(a) + zeros(size(b));
other = double(b) + zeros(size(a));
if wantMax, take = other > m; else, take = other < m; end
take = take | (m ~= m);
m(take) = other(take);
end

function [m, idx] = extremeAlong(x, dim, wantMax)
% Knock-out rounds: rows 1-2, 3-4, ... are compared and the winner moves on.
% Ties keep the earlier row, so idx is the first position, as in MATLAB.
[A, sz, order] = toColumns(x, dim);
n = size(A, 1);
I = (1:n)' * ones(1, size(A, 2));
A(A ~= A) = -inf * (2 * wantMax - 1);      % NaN never wins
while size(A, 1) > 1
    if numel(1:2:size(A, 1)) * 2 > size(A, 1)
        A(end + 1, :) = -inf * (2 * wantMax - 1);  I(end + 1, :) = 0;
    end
    a = A(1:2:end, :);  b = A(2:2:end, :);
    ia = I(1:2:end, :);  ib = I(2:2:end, :);
    if wantMax, take = b > a; else, take = b < a; end
    a(take) = b(take);  ia(take) = ib(take);
    A = a;  I = ia;
end
m = fromColumns(A, sz, order);
idx = fromColumns(I, sz, order);
end

function tf = myAll(x)
tf = mySum(~x(:)) == 0;
end

function y = myAbs(x)
y = x;  neg = y < 0;  y(neg) = -y(neg);
end

function r = myFloor(x)
% Adding and then removing 2^52 rounds a double to the nearest whole number
% (the fraction bits fall off the end); one step down fixes the round-ups.
C = 4503599627370496;                % 2^52: from here on every double is whole
x = double(x);  r = x;
small = myAbs(x) < C;                % also false for Inf and NaN
pos = small & x >= 0;  neg = small & x < 0;
r(pos) = (x(pos) + C) - C;
r(neg) = (x(neg) - C) + C;
up = r > x;
r(up) = r(up) - 1;
end

function m = myMod(a, b)
% a - floor(a/b) * b, the sign of b, as MATLAB's mod
m = a - myFloor(a ./ b) .* b;
zero = (b + zeros(size(a))) == 0;
m(zero) = a(zero);
end

function idx = myFind(c, k)
% Positions of the non-zero entries (a row for a row input, else a column);
% myFind(c, k) keeps the first k.
lin = 1:numel(c);
lin = lin(c(:)' ~= 0);
if nargin > 1 && numel(lin) > k, lin = lin(1:k); end
if size(c, 1) ~= 1, lin = lin'; end
idx = lin;
end

function [r, cc] = myFind2(B)
% Row and column of every non-zero entry, column by column.
lin = myFind(B(:));
r = myMod(lin - 1, size(B, 1)) + 1;
cc = (lin - r) / size(B, 1) + 1;
end

function y = mySqrt(x)
% Square root by Newton's method, y <- (y + x/y)/2, after scaling x by powers
% of 4 into [1/4, 4] so that six steps reach full double precision.
x = double(x);  y = zeros(size(x));
pos = x > 0 & x < inf;
m = x(pos);  s = ones(size(m));
for f = [2^128, 2^32, 2^8, 4]
    while ~isempty(m(m > f))
        b = m > f;  m(b) = m(b) / f;  s(b) = s(b) * mySqrtPow2(f);
    end
    while ~isempty(m(m < 1 / f))
        b = m < 1 / f;  m(b) = m(b) * f;  s(b) = s(b) / mySqrtPow2(f);
    end
end
r = (1 + m) / 2;
for it = 1:6, r = (r + m ./ r) / 2; end
y(pos) = r .* s;
y(x == inf) = inf;
y(x < 0 | x ~= x) = NaN;
end

function r = mySqrtPow2(f)
% Square root of an even power of two: 2^128 -> 2^64, ..., 4 -> 2.
r = 1;
while r * r < f, r = r * 2; end
end

function y = myExp(x)
% e^x = 2^k e^r with k the nearest whole number to x / ln 2 and |r| <= ln(2)/2;
% e^r from its Taylor series 1 + r + r^2/2! + ... (20 terms). ln 2 is split into
% a short head (exact when multiplied by k) and a tail, so r keeps every digit.
LN2 = 0.6931471805599453;  LN2_HEAD = 0.6931471803691238;  LN2_TAIL = 1.9082149292705877e-10;
x = double(x);
k = myFloor(x / LN2 + 0.5);
r = (x - k * LN2_HEAD) - k * LN2_TAIL;
term = ones(size(x));  y = term;
for n = 1:20
    term = term .* r / n;
    y = y + term;
end
y = y .* 2 .^ k;
y(x == inf) = inf;  y(x == -inf) = 0;  y(x ~= x) = NaN;
end

function y = myLog(x)
% ln x = e ln 2 + ln m, where x = m 2^e and m is near 1. For ln m the series
% ln m = 2 (z + z^3/3 + z^5/5 + ...), z = (m - 1)/(m + 1), |z| < 0.18 (20 terms).
LN2 = 0.6931471805599453;  SQRT2 = 1.4142135623730951;
x = double(x);  y = zeros(size(x));
pos = x > 0 & x < inf;
m = x(pos);  e = zeros(size(m));
for k = [64, 16, 4, 1]
    f = 2 ^ k;
    while ~isempty(m(m >= f))
        b = m >= f;  m(b) = m(b) / f;  e(b) = e(b) + k;
    end
    while ~isempty(m(m < 1 / f))
        b = m < 1 / f;  m(b) = m(b) * f;  e(b) = e(b) - k;
    end
end
b = m < 1;    m(b) = m(b) * 2;  e(b) = e(b) - 1;   % now 1 <= m < 2
b = m > SQRT2; m(b) = m(b) / 2;  e(b) = e(b) + 1;  % now 0.71 < m <= 1.41
z = (m - 1) ./ (m + 1);  z2 = z .* z;
t = z;  s = z;
for n = 3:2:41
    t = t .* z2;
    s = s + t / n;
end
y(pos) = e * LN2 + 2 * s;
y(x == 0) = -inf;  y(x == inf) = inf;
y(x < 0 | x ~= x) = NaN;
end

function y = myLog10(x)
y = myLog(x) / 2.302585092994046;    % ln 10
end

function a = myAtan2(y, x)
% Angle of the point (x, y) in (-pi, pi]. atan(t) for 0 <= t <= 1 comes from its
% series after halving the angle twice, t <- t / (1 + sqrt(1 + t^2)); the octant
% is then restored from the signs and the sizes of x and y.
y = double(y) + zeros(size(x));  x = double(x) + zeros(size(y));
ax = myAbs(x);  ay = myAbs(y);
big = myMax(ax, ay);  small = myMin(ax, ay);
t = zeros(size(x));
nz = big > 0;
t(nz) = small(nz) ./ big(nz);
for h = 1:2, t = t ./ (1 + mySqrt(1 + t .* t)); end
t2 = t .* t;  term = t;  s = t;
for n = 3:2:41
    term = -term .* t2;
    s = s + term / n;
end
a = 4 * s;
a(ay > ax) = PI() / 2 - a(ay > ax);
a(x < 0) = PI() - a(x < 0);
a(y < 0) = -a(y < 0);
end

function [s, c] = mySinCos(th)
% Sine and cosine from their Taylor series, after moving th into [-pi, pi].
th = double(th);
r = th - 2 * PI() * myFloor(th / (2 * PI()) + 0.5);
r2 = r .* r;
s = r;  c = ones(size(r));
ts = r;  tc = ones(size(r));
for n = 1:15
    ts = -ts .* r2 / ((2 * n) * (2 * n + 1));
    tc = -tc .* r2 / ((2 * n - 1) * (2 * n));
    s = s + ts;  c = c + tc;
end
end

function v = myNorm(x)
v = mySqrt(mySum(x(:) .^ 2));
end

function x = mySolve(A, b)
% Solves A x = b by Gaussian elimination with partial pivoting.
n = size(A, 1);  M = [double(A), double(b)];
for k = 1:n
    [~, p] = myMax(myAbs(M(k:n, k)));
    p = p + k - 1;
    M([k p], :) = M([p k], :);
    for i = k + 1:n
        M(i, :) = M(i, :) - M(i, k) / M(k, k) * M(k, :);
    end
end
x = zeros(n, size(b, 2));
for k = n:-1:1
    x(k, :) = (M(k, n + 1:end) - M(k, k + 1:n) * x(k + 1:n, :)) / M(k, k);
end
end

function v = myLinspace(a, b, n)
v = a + (0:n - 1) * (b - a) / (n - 1);
v(end) = b;
end

function [X, Y] = myMeshgrid(u, v)
% X repeats the row u for every entry of v; Y repeats the column v.
X = ones(numel(v), 1) * u(:)';
Y = v(:) * ones(1, numel(u));
end

function B = myRepmat(A, r, c)
B = A(myMod(0:r * size(A, 1) - 1, size(A, 1)) + 1, ...
      myMod(0:c * size(A, 2) - 1, size(A, 2)) + 1);
end

function I = myEye(n)
I = zeros(n);  I(1:n + 1:end) = 1;
end

function D = keepDiagonal(A)
% A with everything off the main diagonal set to zero
n = size(A, 1);  D = zeros(n);  D(1:n + 1:end) = A(1:n + 1:end);
end

function U = upperMask(n)
% true above the main diagonal of an n x n matrix
U = (1:n)' < (1:n);
end
