content = open('train.py', 'r', encoding='utf-8').read()

# Fix learning rate
content = content.replace(
    '"learning_rate" : 1e-4,    # Adam optimizer starting learning rate',
    '"learning_rate" : 1e-3,    # Adam optimizer starting learning rate'
)

# Fix ReduceLROnPlateau patience
content = content.replace(
    'patience = 5,            # Wait 5 epochs before reducing',
    'patience = 10,           # Wait 10 epochs before reducing'
)

# Fix EarlyStopping patience
content = content.replace(
    'patience             = 15,       # Stop after 15 epochs without improvement',
    'patience             = 25,       # Stop after 25 epochs without improvement'
)

# Fix ReduceLROnPlateau factor
content = content.replace(
    'factor   = 0.1,          # New LR = current LR × 0.1',
    'factor   = 0.5,          # New LR = current LR × 0.5'
)

# Fix num_epochs
content = content.replace(
    '"num_epochs"    : 100,     # Max epochs (EarlyStopping will likely stop earlier)',
    '"num_epochs"    : 150,     # Max epochs (EarlyStopping will likely stop earlier)'
)

open('train.py', 'w', encoding='utf-8').write(content)
print("train.py fixed")
