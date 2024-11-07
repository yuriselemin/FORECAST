from django.db import models

class Stock(models.Model):
    ticker = models.CharField(max_length=10)
    start_date = models.DateField()
    end_date = models.DateField()
    model_type = models.CharField(max_length=20)
    mse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.ticker}: {self.start_date} - {self.end_date}"