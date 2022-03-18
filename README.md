# Usage

## Creating the StateManager

```
s = StateManager('weights/', 'messages/')
```

The weights folder should contain numbered weights files.
The weights folder and the messages folder may be the same.

## Loading the Most Recent Weights

```
s.give_most_recent_weights(model, device)
```

You can determine the epoch that was just loaded like this:

```
s.determine_most_recent_state().epoch
```

## Saving the state of the model

```
s.commit(model)
```

You can also save a message as follows:

```
s.commit(model, message)
```

Messages are stored as plain text files.

## TODO

* Support using dates instead of numbers before the suffix.