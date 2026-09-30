from django.test import SimpleTestCase


class ChunkRecoveryScheduleTests(SimpleTestCase):
    def test_application_scheduler_is_explicit_and_independent_of_graph_features(self):
        from aquillm.celery_schedules import application_maintenance_schedule
        self.assertEqual(application_maintenance_schedule(enabled=False), {})
        entry = application_maintenance_schedule(enabled=True)['document-chunk-publication-recovery']
        self.assertEqual(entry['task'], 'apps.documents.tasks.chunk_recovery.recover_chunk_publications')
        self.assertEqual(entry['schedule'], 60)
        self.assertEqual(entry['kwargs'], {'limit': 25})
        self.assertEqual(entry['options']['queue'], 'celery')
