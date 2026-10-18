"""Optional certificate jobs; failures never enter projection staging retries."""

import logging
from threading import Event, Lock, Thread
from uuid import UUID

from celery import current_app, shared_task
from celery.local import Proxy
from kombu import Producer
from kombu.transport.redis import Channel as RedisChannel
from redis.backoff import NoBackoff
from redis.retry import Retry

from .membership_certification import certify_ready_membership
from .runtime import load_projection_runtime_settings

TASK_NAME = "apps.knowledge_graph.projection.membership_certificate_tasks.certify_membership"
_DISPATCH_SECONDS = 2.0
_DISPATCH_SLOT = Lock()
logger = logging.getLogger(__name__)


class _CertificateChannel(RedisChannel):
    def _connparams(self, asynchronous=False):
        params = super()._connparams(asynchronous=asynchronous)
        params.update(socket_connect_timeout=1, socket_timeout=1,
                      retry_on_timeout=False, retry_on_error=[],
                      retry=Retry(NoBackoff(), 0))
        cancelled = self.connection.client.transport_options.get("certificate_cancelled")
        if cancelled is not None:
            native_class = params.get("connection_class") or self.connection_class

            class CancellableConnection(native_class):
                def _connect(self):
                    if cancelled.is_set():
                        raise TimeoutError("membership_certificate_dispatch_cancelled")
                    sock = super()._connect()
                    # redis-py performs blocking getaddrinfo inside _connect.
                    # Once it returns, fence authentication/publication on timeout.
                    if cancelled.is_set():
                        sock.close()
                        raise TimeoutError("membership_certificate_dispatch_cancelled")
                    return sock

            params["connection_class"] = CancellableConnection
        return params


@shared_task(name=TASK_NAME, max_retries=0, acks_late=False,
             reject_on_worker_lost=False, soft_time_limit=295, time_limit=300,
             ignore_result=True)
def certify_membership(projection_id):
    try:
        identifier = UUID(projection_id)
        if str(identifier) != projection_id:
            return {"certified": False, "failure_code": "invalid_projection_id"}
        return {"certified": certify_ready_membership(projection_id=identifier), "failure_code": None}
    except TimeoutError:
        return {"certified": False, "failure_code": "certificate_timeout"}
    except Exception:
        return {"certified": False, "failure_code": "certificate_failed"}


def _publish_certificate(*, app, settings, projection_id, cancelled):
    # Dedicated unpooled Redis connection: no pool-acquisition wait or retries.
    with app.connection_for_write(connect_timeout=1) as connection:
        if connection.transport.driver_type != "redis":
            logger.info("obs.knowledge_graph.membership_certificate_dispatch_unsupported_broker")
            return None
        connection.transport.Channel = _CertificateChannel
        connection.transport_options = {
            **connection.transport_options, "socket_connect_timeout": 1,
            "socket_timeout": 1, "retry_on_timeout": False,
            "max_retries": 0,
            "certificate_cancelled": cancelled,
        }
        connection.ensure_connection(max_retries=0)
        # A resolver delayed past caller cancellation must not publish later.
        if cancelled.is_set():
            return False
        with Producer(connection) as producer:
            if cancelled.is_set():
                return False
            certify_membership.apply_async(args=(str(projection_id),),
                producer=producer, queue=settings.projection_queue,
                retry=False, expires=300)
    return True


def dispatch_membership_certificate(projection_id):
    try:
        settings = load_projection_runtime_settings()
        if not (getattr(settings, "membership_certificate_enabled", False)
                or getattr(settings, "membership_certificate_issuance_enabled", False)):
            return False
        # Resolve Celery's thread-local proxy on the calling worker thread.
        app = current_app._get_current_object() if isinstance(current_app, Proxy) else current_app
        if app.conf.task_always_eager:
            logger.info("obs.knowledge_graph.membership_certificate_dispatch_eager_skipped")
            return False
        if not _DISPATCH_SLOT.acquire(blocking=False):
            logger.warning("obs.knowledge_graph.membership_certificate_dispatch_busy")
            return False
        completed, cancelled = Event(), Event()
        result = []

        def publish():
            try:
                result.append(_publish_certificate(app=app, settings=settings,
                    projection_id=projection_id, cancelled=cancelled))
                if result[0] is False and not cancelled.is_set():
                    logger.warning("obs.knowledge_graph.membership_certificate_dispatch_failed")
            except Exception:
                result.append(False)
                logger.warning("obs.knowledge_graph.membership_certificate_dispatch_failed")
            finally:
                # Physical ownership ends only after the helper's connection and
                # producer context managers have finished closing, even on timeout.
                _DISPATCH_SLOT.release()
                completed.set()

        helper = Thread(target=publish, name="membership-certificate-dispatch", daemon=True)
        try:
            helper.start()
        except Exception:
            _DISPATCH_SLOT.release()
            raise
        if not completed.wait(_DISPATCH_SECONDS):
            cancelled.set()
            logger.warning("obs.knowledge_graph.membership_certificate_dispatch_timeout")
            return False
        return bool(result and result[0])
    except Exception:
        logger.warning("obs.knowledge_graph.membership_certificate_dispatch_failed")
        return False
