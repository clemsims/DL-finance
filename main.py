from data import t_student_drift
from features import build_features
from models import predict_model, train_model
from visualization import plot

xi, mu = t_student_drift.generate_timeseries()
X_train, X_test, y_train, y_test = build_features.prepare_train(xi, mu)
model = train_model.create_model()

train_model.train_model(model, X_train, y_train, X_test, y_test)

predict_model.predict_mu(model)

# BUG: memory allocation issues, N too big, reduce its size
predict_model.generalization_evaluation(model)
