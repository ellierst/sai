clc; clear; close all;

%% Зчитування навчальних даних
run('p_Read_Vector_1.m');

%% Параметри
QV1      = 16;          % шар 1 — фіксований
QV2_base = 12;          % базове значення шару 2
factors  = [0.5, 1.0, 1.5];   % -50%, 0%, +50%
m_runs   = 7;

results  = zeros(3, m_runs);
epochs_t = zeros(3, m_runs);

for f = 1:3
    QV2 = round(QV2_base * factors(f));
    fprintf('\n%s\n', repmat('-', 1, 55));
    fprintf('QV2 = %d нейронів (%+.0f%%)\n', QV2, (factors(f)-1)*100);
    fprintf('%s\n', repmat('-', 1, 55));

    for k = 1:m_runs
        net = newff(P, T, [QV1 QV2], {'logsig', 'tansig'});
        net.divideFcn             = 'dividetrain';
        net.trainFcn              = 'trainbfg';
        net.performFcn            = 'mse';
        net.trainParam.epochs     = 1500;
        net.trainParam.goal       = 1e-3;
        net.trainParam.time       = 300;
        net.trainParam.min_grad   = 1e-10;
        net.trainParam.showWindow = false;

        tic;
        [~, tr] = train(net, P, T, [], [], [], []);
        t = toc;

        results(f, k)  = t;
        epochs_t(f, k) = tr.epoch(end);

        fprintf('  Спроба %d: %.2f сек \n', ...
            k, t, tr.epoch(end), tr.perf(end));
    end
end

%% Статистичний аналіз
t_gamma = 2.45;   % при m=7, γ=0.95
delta1  = 0.1;

fprintf('\n%s\n', repmat('=', 1, 72));
fprintf('%-20s | %-9s | %-9s | %-9s | %-7s\n', ...
    'Конфігурація', 'tc (с)', 'σ', 'δ', 'm1');
fprintf('%s\n', repmat('-', 1, 72));

labels_box = {'-50% (QV2=6)', '0% (QV2=12)', '+50% (QV2=18)'};
labels_fmt = {'-50%% (QV2=6)', ' 0%% (QV2=12)', '+50%% (QV2=18)'};

stats = zeros(3, 4);

for f = 1:3
    t     = results(f, :);
    tc    = mean(t);
    sigma = sqrt(sum((t - tc).^2) / m_runs);    % формула А.2
    s     = sigma * sqrt(m_runs / (m_runs - 1));
    delta = t_gamma * s / sqrt(m_runs);          % формула А.3
    m1    = ceil((t_gamma * s / delta1)^2);      % формула 3.6

    stats(f,:) = [tc, sigma, delta, m1];

    fprintf('%-20s | %-9.3f | %-9.3f | %-9.3f | %-7d\n', ...
        labels_fmt{f}, tc, sigma, delta, m1);
end
fprintf('%s\n', repmat('=', 1, 72));

%% графік часу навчання по спробах
figure('Position', [100 100 850 480]);
bar(results');
legend({'-50% (QV2=6)', '0% (QV2=12)', '+50% (QV2=18)'}, ...
    'Location', 'NorthEast');
xlabel('Номер випробування (k)');
ylabel('Час навчання t(k), сек');
title('Залежність часу навчання від кількості нейронів у шарі 2 (QV2)');
xticks(1:m_runs);
grid on;