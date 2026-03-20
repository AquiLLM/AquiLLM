"""Gemini API usage tracking model."""
from django.db import models


class GeminiAPIUsage(models.Model):
    """Model to track Gemini API usage and costs"""
    timestamp = models.DateTimeField(auto_now_add=True)
    operation_type = models.CharField(max_length=100, help_text="Type of operation (e.g., 'OCR', 'Handwritten Notes')")
    input_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField(default=0)
    cost = models.DecimalField(max_digits=10, decimal_places=6, default=0)

    INPUT_COST_PER_1K = 0.0005
    OUTPUT_COST_PER_1K = 0.0015

    class Meta:
        app_label = 'apps_platform_admin'
        db_table = 'aquillm_geminiapiusage'
        verbose_name = "Gemini API Usage"
        verbose_name_plural = "Gemini API Usage"
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.operation_type} at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    @classmethod
    def calculate_cost(cls, input_tokens, output_tokens):
        """Calculate cost based on token usage"""
        input_cost = (input_tokens / 1000) * cls.INPUT_COST_PER_1K
        output_cost = (output_tokens / 1000) * cls.OUTPUT_COST_PER_1K
        return input_cost + output_cost

    @classmethod
    def log_usage(cls, operation_type, input_tokens, output_tokens):
        """Log API usage and return the cost"""
        cost = cls.calculate_cost(input_tokens, output_tokens)
        usage = cls.objects.create(
            operation_type=operation_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost
        )
        return usage

    @classmethod
    def get_total_stats(cls):
        """Get aggregated usage statistics"""
        from django.db.models import Sum, Count

        stats = cls.objects.aggregate(
            total_input_tokens=Sum('input_tokens'),
            total_output_tokens=Sum('output_tokens'),
            total_cost=Sum('cost'),
            api_calls=Count('id')
        )
        return stats
