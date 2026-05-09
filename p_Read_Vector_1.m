% p_Read_Vector_1.m
% Зчитує BMP-зображення навчальної вибірки та формує матриці P і T

classFile = 'Neuro_Data/Neuro_Train/Class_Name_ty1.txt';
trainFile = 'Neuro_Data/Neuro_Train/File_Name_tx1.txt';

%% Зчитування класів
fid = fopen(classFile, 'r');
if fid == -1
    error('Не вдалося відкрити файл: %s', classFile);
end
classData  = textscan(fid, '%d %s');
fclose(fid);

classIDs   = classData{1};   % [1; 2; 3; 4]
classNames = classData{2};   % {'L', 'м', 'Q', 'М'}
numClasses = length(classIDs);

%% Зчитування навчальної вибірки
fid = fopen(trainFile, 'r');
if fid == -1
    error('Не вдалося відкрити файл: %s', trainFile);
end
trainData = textscan(fid, '%d %s');
fclose(fid);

trainLabels = trainData{1};   % номери класів (стовпець)
trainFiles  = trainData{2};   % імена файлів (без розширення)
numTrain    = length(trainLabels);

%% Формування матриць P та T
imgSize = 16 * 16;   % 256 пікселів на зображення

P = zeros(imgSize, numTrain);       % кожен стовпець — один вектор X
T = zeros(numClasses, numTrain);    % one-hot цільові вектори

for n = 1:numTrain
    imgPath = ['Neuro_Data/Neuro_Train/' trainFiles{n} '.bmp'];

    if ~exist(imgPath, 'file')
        error('Не знайдено зображення: %s', imgPath);
    end

    img = imread(imgPath);

    % Перетворення у відтінки сірого (якщо кольорове)
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    % Приведення до розміру 16×16 (на випадок невідповідності)
    if ~isequal(size(img), [16 16])
        img = imresize(img, [16 16]);
    end

    % Нормалізація пікселів: [0, 255] → [0, 1]
    P(:, n) = double(img(:)) / 255;

    % Цільовий вектор: one-hot encoding
    T(trainLabels(n), n) = 1;
end

fprintf('Зчитано зображень: %d\n', numTrain);
fprintf('Розмір P: %d × %d\n',    size(P,1), size(P,2));
fprintf('Розмір T: %d × %d\n',    size(T,1), size(T,2));