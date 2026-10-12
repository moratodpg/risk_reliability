# Synthetic observation records

| File | Column | Unit and observation level | Number |
|---|---|---|---:|
| alarm_trials.csv | alarm_delay_s | Seconds from ignition to notification; one independent controlled alarm trial | 80 |
| response_drills.csv | shared_response_s | Seconds after notification for the common global response component; one independent drill | 60 |
| walking_trials.csv | free_speed_m_s | Meters/second; one different participant in a free-walking test | 100 |
| training_followup.csv | trained_to_baseline_ratio | Dimensionless paired response ratio; one independent matched drill pair | 12 |

`record_id`/`pair_id` identifies an independent observation, not a time series.
`synthetic` is a provenance flag, not a model predictor.

For the exercise, observations are complete (no censoring or missing records),
representative of the baseline scenario, stationary and mutually independent.

The response observation is the *shared* drill component P. The separate
person-specific residual J_i ~ Uniform(0,12) seconds is supplied in the model;
do not add it to the fitted historical observations and then count it twice.
The speed data represent unimpeded travel; door congestion is handled separately.

The follow-up file is OPTIONAL and must not be integrated with baseline response
observations. If used, the supplied likelihood is
log(ratio_j) ~ Normal(theta, 0.12^2), where theta is the unknown log training
multiplier. The prior is Normal(log(0.70), 0.20^2). The 0.12 is observation noise
on paired log ratios, not person-to-person speed or response variability.

Select distributions for alarm delay, response, and speed. Starter
lognormal moment fits are provisional defaults, not the expected answers.
