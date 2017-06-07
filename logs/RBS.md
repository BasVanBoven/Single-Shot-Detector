## Context

In order to translate the problem of window classification to real-world action classification, we have to implement some business rules which determine if the system should send an alert. We send one frame per camera per second to the object detector, and can thus also send one request per camera per second to the action classifier (this request contains information about the last five frames, and thus we also have to implement some graceful startup routine).


## Business rules, action classifier

The output from the action classifier combined with localisation information extracted from the object detector (we use the average center point of the wheelbase for this) can be used to label a situation to one of the following categories:

  - Positive classification within zone: true positive
  - Positive classification out of zone: negative
  - Negative classification within zone: negative
  - Negative classification out of zone: negative

Now, we can count the number of true positives over a certain interval. When this number of true positives exceeds a certain threshold, the alert gets sent.


## Business rules, object detector

For the business rules related to crouching and earth rod activity, we only use the classification and localisation information of the object detector (again, taking the center point of the action as localisation). Now, we can again count the number of true positives over a certain interval. Crouching and earth rod activity each have an independent counter associated with them. A difference to the business rules of the sequence processor is that we are not confined to one crouching and earth rod detection per frame, so if multiple crouching and earth rod detections are in the same frame, the counter can increment quicker. This is reasonable as multiple crouching and earth rod detections point to a higher likelyhood of illegal digging activity.


## Variables

In the end, this leaves us with the following four parameters to tune:

- *Interval_length*: the time period over which to count the positive clasifications.
- *Threshold_excavator*: minimum value of the excavator counter before an alert gets sent.
- *Threshold_crouching*: minimum value of the crouching counter before an alert gets sent.
- *Threshold_earthrod*: minimum value of the earth rod counter before an alert gets sent.


## Latency

Latency of the proposed solution depends on the thresholds, but is 5 + *Interval_length* at most.


## In case of problems

If all of this fails, we could additionally manipulate the counter with a rule-based system designed to weed out easy to discern situations like driving and an empty frame. However, this would likely point to a broken action classifier, so it would probably be better to fix these problems there.