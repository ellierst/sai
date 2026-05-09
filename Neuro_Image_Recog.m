clc; clear; close all;

%% Параметри
recognDir  = 'Neuro_Data/Neuro_Recogn/';
classNames = {'L', 'м', 'Q', 'М'};
numClasses = 4;

testFiles = {
    'L1_damaged',  'L1_rotated',  'L1_scaled',  'L1_shifted', ...
    'м1_damaged',  'м1_rotated',  'м1_scaled',  'м1_shifted', ...
    'Q1_damaged',  'Q1_rotated',  'Q1_scaled',  'Q1_shifted', ...
    'M1_damaged',  'M1_rotated',  'M1_scaled',  'M1_shifted'
};

trueLabels = [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4];

goalValues = [1e-2, 1e-3, 1e-4];

for g = 1:length(goalValues)
    fileName = sprintf('net_mse%s', ...
        strrep(sprintf('%.0e', goalValues(g)), '-', 'm'));

    if ~exist([fileName '.mat'], 'file')
        fprintf('\n[!] Не знайдено: %s.mat\n', fileName);
        continue;
    end

    load([fileName '.mat'], 'net_trained');

    fprintf('\n%s\n', repmat('=', 1, 75));
    fprintf('  mse = %.0e\n', goalValues(g));
    fprintf('%s\n', repmat('=', 1, 75));
    fprintf('%-16s | %-5s | %-10s | %s\n', 'Файл', 'Клас', 'Результат', 'Вектор Y');
    fprintf('%s\n', repmat('-', 1, 75));

    correctCount = 0;
    allY = zeros(numClasses, length(testFiles));

    for n = 1:length(testFiles)
        imgPath = [recognDir testFiles{n} '.bmp'];

        if ~exist(imgPath, 'file')
            fprintf('[!] Не знайдено: %s\n', imgPath);
            continue;
        end

        img = imread(imgPath);
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        if ~isequal(size(img), [16 16])
            img = imresize(img, [16 16]);
        end

        imgVec = double(img(:)) / 255;
        Y = sim(net_trained, imgVec);
        allY(:, n) = Y;

        [~, recognClass] = max(Y);
        isCorrect = (recognClass == trueLabels(n));
        if isCorrect
            correctCount = correctCount + 1;
        end

        marker = '';
        if ~isCorrect
            marker = ' <--';
        end

        yStr = sprintf('%.3f ', Y);
        fprintf('%-16s | %-5d | %-10s | [%s]%s\n', ...
            testFiles{n}, recognClass, classNames{recognClass}, yStr, marker);
    end

    accuracy = correctCount / length(testFiles) * 100;
    fprintf('\nТочність розпізнавання: %d/16 (%.1f%%)\n', correctCount, accuracy);

    figure('Position', [100 100 1000 420]);
    bar(allY');
    set(gca, 'XTick', 1:length(testFiles), ...
             'XTickLabel', testFiles, ...
             'XTickLabelRotation', 45);
    legend(classNames, 'Location', 'NorthEast');
    title(sprintf('mse = %.0e', goalValues(g)));
    ylabel('Активація нейрона виходу');
    grid on;
end