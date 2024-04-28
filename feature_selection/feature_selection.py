# File Handling Imports. 
import csv
import os
import numpy as np

# Correlation Matrix.
import matplotlib.pyplot as plt
import seaborn as sns

# Annova and Chi.
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression, mutual_info_classif
from scipy import stats

X = []
Y = []
X_test = []
y_test = []

def train(folder):
    """Get X and Y features from csv files."""
    global X, Y
    for filename in os.listdir(folder):
        if filename.startswith('drowsy'):
            ylabel = 1
        else:
            ylabel = 0
        with open(os.path.join(folder,filename),'r') as f:
            re = csv.reader(f,delimiter=',')
            for row in re:
                a = []
                for x in row:
                    a.append(float(x))
                X.append(a)
                Y.append(ylabel)


train(r'Data2\Fold1')
train(r'Data2\Fold2')
train(r'Data2\Fold3')
train(r'Data2\Fold4')

print('file part done!\tX_train = {}'.format(len(X)))

feature_labels = ['MR', 'eye_closed', 'freq', 'perclo', 'eye_circ', 'pupil', 'eyebrow', 'moer', 'ear']
X = np.array(X)
Y = np.array(Y)
# Correlation Matrix--------------------------------------------------------------------------------------
correlation_matrix = np.corrcoef(X, rowvar=False)
sns.set(style="white")  # Set the style of the visualization
plt.figure(figsize=(8, 6))  # Set the size of the figure
# Create a heatmap with a color map
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=feature_labels, yticklabels=feature_labels)
plt.title("Pearson Correlation Matrix")
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------
# Chi score <-- how well the feature and target relate to each other.
print('CHI SCORE------')
selector = SelectKBest(chi2, k = len(feature_labels))
X_chi = X - X.min() + 1    # chi2 only works with non-negative data.
x_selected = selector.fit_transform(X_chi, Y)
indices = selector.get_support(indices=True)
chi = {}
for name, val in zip([feature_labels[i] for i in indices], selector.scores_[indices]):
    chi[name] = val
for i,j in sorted(chi.items(), key=lambda t: t[1]):
    print('%-10s' % i, end = "")
    print('\t{}'.format(j))
# #----------------------------------------------------------------------------------------------------------------------------------  
# ANOVA Score -- tells us how well that feature discriminates between classes.
print('ANOVA SCORE------')
selector = SelectKBest(f_classif, k = len(feature_labels))
x_selected = selector.fit_transform(X, Y)
indices = selector.get_support(indices=True)
anno = {}
for name, val in zip([feature_labels[i] for i in indices], selector.scores_[indices]):
    anno[name] = val
for i,j in sorted(anno.items(), key=lambda t: t[1]):
    print('%-10s' % i, end = "")
    print('\t{}'.format(j))
#----------------------------------------------------------------------------------------------------------------------------------
# Spearman Score -- like correlation matrix. Pval tell us how well it does with alternative score
print('SPEARMAN SCORE------')
correlation_matrix = stats.spearmanr(X, axis=0, alternative='greater')
fig, axis = plt.subplots(2,2,figsize = (15,5))
sns.set(style="white")  # Set the style of the visualization
sns.heatmap(correlation_matrix[0], annot=True, cmap="coolwarm", fmt=".2f", xticklabels=feature_labels, yticklabels=feature_labels, ax = axis[0,0])
axis[0,0].set_title("Correlation Matrix")
sns.heatmap(correlation_matrix[1], annot=True, cmap="coolwarm", fmt=".2f", xticklabels=feature_labels, yticklabels=feature_labels, ax = axis[0,1])
axis[0,1].set_title("Greater Hypo P-Val")

correlation_matrix = stats.spearmanr(X, axis=0, alternative='less')
sns.heatmap(correlation_matrix[1], annot=True, cmap="coolwarm", fmt=".2f", xticklabels=feature_labels, yticklabels=feature_labels, ax = axis[1,0])
axis[1,0].set_title("Less Hypo P-Val")

correlation_matrix = stats.spearmanr(X, axis=0)
sns.heatmap(correlation_matrix[1], annot=True, cmap="coolwarm", fmt=".2f", xticklabels=feature_labels, yticklabels=feature_labels, ax = axis[1,1])
axis[1,1].set_title("2-Sided P-Val")

plt.tight_layout()
plt.suptitle("Spearman")
fig.set_size_inches(12, 11)
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------  
# KENDALLS RANK
print('Kendalls Rank SCORE------')
selector = SelectKBest(f_regression, k = len(feature_labels))
x_selected = selector.fit_transform(X, Y)
indices = selector.get_support(indices=True)
kendall = {}
for name, val in zip([feature_labels[i] for i in indices], selector.scores_[indices]):
    kendall[name] = val
for i,j in sorted(kendall.items(), key=lambda t: t[1]):
    print('%-10s' % i, end = "")
    print('\t{}'.format(j))
#----------------------------------------------------------------------------------------------------------------------------------  
# MUTAL INFORMATION
print('Mutual Information Rank SCORE------')
selector = mutual_info_classif(X,Y)
mi = {}
for name, val in enumerate(selector):
    mi[feature_labels[name]] = val
# for i,j in sorted(x.items(), key=lambda t: t[1]):
#     print('%-10s' % i, end = "")
#     print('\t{}'.format(j))
#----------------------------------------------------------------------------------------------------------------------------------
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=1) # no. of classes - 1
lda.fit(X, Y)
plt.figure(figsize=(10, 6))
sns.heatmap(lda.coef_, annot=True, cmap='coolwarm', xticklabels=feature_labels, yticklabels=['Discriminant 1'])
plt.suptitle('LDA Feature Coefficients')
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------
# PCA
from sklearn.decomposition import PCA

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)
fig, axis = plt.subplots(1, 2, figsize=(12, 5))
axis[0].plot(np.cumsum(pca.explained_variance_ratio_))
axis[0].set_xlabel('Number of Principal Components')
axis[0].set_ylabel('Cumulative Explained Variance Ratio')
axis[0].set_title('Explained Variance Ratio by Principal Components')

# Get the absolute values of the loading vectors (components_)
loading_vectors = np.abs(pca.components_)
# Visualize the importance of original features in the principal components
im = axis[1].imshow(loading_vectors, cmap='viridis', aspect='auto')

# Annotate the plot with feature labels
fig.colorbar(im, ax=axis[1], orientation='vertical', label='Absolute Loading Value')
axis[1].set_xticks(range(len(feature_labels)), feature_labels, rotation=45, ha='right')
axis[1].set_yticks(range(len(feature_labels)), [f'Principal Component {i}' for i in range(len(feature_labels))])
axis[1].set_title('Importance of Original Features in Principal Components')

# Annotate each cell with the corresponding loading value
for i in range(len(loading_vectors)):
    for j in range(len(feature_labels)):
        axis[1].text(j, i, f'{loading_vectors[i, j]:.2f}', ha='center', va='center', color='white' if loading_vectors[i, j] > 0.5 else 'black')
plt.tight_layout()
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------
# Plotting

# List of feature labels
features = feature_labels

# Extract scores from each dictionary
scores_1 = list(chi.values())
scores_2 = list(anno.values())
scores_3 = list(kendall.values())
scores_4 = list(mi.values())

# Plotting
bar_width = 0.4
bar_positions_1 = np.linspace(0, len(features)*1.5, len(features))
bar_positions_2 = [pos + bar_width for pos in bar_positions_1]
bar_positions_3 = [pos + bar_width for pos in bar_positions_2]
bar_positions_4 = [pos + bar_width for pos in bar_positions_3]

plt.figure(figsize=(10, 6)) 
plt.bar(bar_positions_1, scores_1, width=bar_width, label='Chi', color = 'red')
plt.bar(bar_positions_2, scores_2, width=bar_width, label='Anova', color = 'blue')
plt.bar(bar_positions_3, scores_3, width=bar_width, label='Kendall', color = 'yellow')
plt.bar(bar_positions_4, scores_4, width=bar_width, label='Mutual Info', color = 'orange')

plt.xlabel('Features')
plt.ylabel('Scores (Log)')
plt.title('Feature Selection Scores')
plt.yscale('log')
plt.xticks((bar_positions_1 + bar_positions_2 + bar_positions_3) / 3, features)
plt.legend()

plt.show()
