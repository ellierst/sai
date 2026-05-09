clc; clear; close all;

%% s1 - Зчитування векторів
p_Read_Vector_1;

%% s2 - Параметри мережі
QV1 = 16;
QV2 = 12;

goalValues = [1e-2, 1e-3, 1e-4];

for g = 1:length(goalValues)

    net = newff(P, T, [QV1 QV2], {'logsig', 'tansig'});
    net.divideFcn           = 'dividetrain';

    %% s4 - Алгоритм навчання
    net.trainFcn = 'trainbfg';

    %% s5 - Параметри навчання
    net.performFcn          = 'mse';
    net.trainParam.epochs   = 1500;
    net.trainParam.time     = 300;
    net.trainParam.goal     = goalValues(g);
    net.trainParam.min_grad = 1e-10;

    fprintf('\nmse = %.0e\n', goalValues(g));

    %% s6 - Навчання
    tic;
    [net_trained, tr] = train(net, P, T, [], [], [], []);
    trainTime = toc;

    fprintf('Час навчання:  %.2f с\n', trainTime);

    if tr.perf(end) <= goalValues(g)
        fprintf('[OK] Ціль досягнута!\n');
    else
        fprintf('[!] Ціль не досягнута\n');
    end

    % Графік навчання
    figure;
    semilogy(tr.epoch, tr.perf, 'b-', 'LineWidth', 2);
    hold on;
    yline(goalValues(g), 'r--', sprintf('%.0e', goalValues(g)), ...
        'LabelHorizontalAlignment', 'left', 'LineWidth', 1.5);
    grid on;
    title(sprintf('mse = %.0e', goalValues(g)));

    %% s7 - Збереження
    fileName = sprintf('net_mse%s', ...
        strrep(sprintf('%.0e', goalValues(g)), '-', 'm'));
    save(fileName, 'net_trained', 'tr', 'trainTime');
    fprintf('Збережено: %s.mat\n', fileName);
end