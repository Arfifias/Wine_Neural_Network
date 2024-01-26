% Διαβάζει τα δεδομένα από το αρχείο csv
wine_data = readtable('winequality-red.csv');

% Διακριτοποίη την ποιότητα (quality) σε κατηγορίες
wine_data.quality_category = discretize(wine_data.quality, [0 5 6 7 10], 'categorical', {'Low', 'Lower Medium', 'Upper Medium',' High'});

% Χωρισμός των δεδομένων σε είσοδο (p) και στόχο (t)
p = table2array(wine_data(:, 1:end-2))'; % Είσοδος: Όλες οι στήλες εκτός των quality και quality_category
t = dummyvar(categorical(wine_data.quality_category))';

% Διαμόρφωση του νευρωνικού δικτύου
hidden_layer_neurons = [12 8];
training_function = 'trainrp';

net = patternnet(hidden_layer_neurons);
net.divideFcn = 'dividerand';
net.trainFcn = training_function;

net.trainParam.goal = 1e-7;
net.trainParam.epochs = 2500;
net.trainParam.show = 1;

net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 40/100;
net.trainParam.max_fail = 15;

% Αρχικοποίση και εκπαίδευση του νευρωνικού δικτύου
net = init(net);
[net, tr] = train(net, p, t);

% Αξιολόγηση του δικτύου
outputs = net(p);
[~, predicted_labels] = max(outputs);
[~, true_labels] = max(t);

% Δημιουργία των κατηγοριών ποιότητας
categories = {'Low', 'Lower Medium', 'Upper Medium', 'High'};
predicted_categories = categories(predicted_labels);
true_categories = categories(true_labels);

% Εμφάνιση του πίνακα σύγχυσης
confusion_matrix = confusionmat(true_categories, predicted_categories);
disp('Πίνακας Σύγχυσης:');
disp(confusion_matrix);