from enum import Enum
import collections

class Status(Enum):
    pending = "PENDING"
    running = "RUNNING"
    success = "SUCCESS"
    failed = "FAILED"

class Job:
    def __init__(self, job_id: str, func):
        self.id = job_id
        self.func = func
        self.status = Status.pending
    
    def run(self):
        self.status = Status.running
        try:
            self.func()
            self.status = Status.success
        except Exception:
            self.status = Status.failed
        

class JobScheduler:

    def __init__(self) -> None:
        self.pending_tasks = collections.deque()
        self.tasks = {}

    def add_job(self, job_id: str, task) -> None:
        """
        Add a new job to the scheduler.

        job_id is unique.
        task is a callable function with no arguments.
        """
        new_job = Job(job_id, task)
        self.pending_tasks.append(new_job)
        self.tasks[job_id] = new_job

    def run_next(self) -> bool:
        """
        Run the next pending job.

        Return True if a job was run.
        Return False if there is no pending job.
        """
        if not self.pending_tasks:
            return False
        self.pending_tasks.popleft().run()
        return True

    def run_all(self) -> None:
        """
        Run all pending jobs in insertion order.
        """
        while self.run_next():
            pass

    def get_status(self, job_id: str) -> str:
        """
        Return the status of the given job.
        """
        return self.tasks[job_id].status

if __name__ == "__main__":
    # def job_a():
    #     print("A finished")

    # def job_b():
    #     print("B finished")

    # scheduler = JobScheduler()

    # scheduler.add_job("A", job_a)
    # scheduler.add_job("B", job_b)

    # print(scheduler.get_status("A"))
    # # PENDING

    # scheduler.run_next()
    # # prints: A finished

    # print(scheduler.get_status("A"))
    # # SUCCESS

    # print(scheduler.get_status("B"))
    # # PENDING

    # scheduler.run_all()
    # # prints: B finished

    # print(scheduler.get_status("B"))
    # # SUCCESS
    def bad_job():
        raise Exception("something went wrong")

    scheduler = JobScheduler()

    scheduler.add_job("bad", bad_job)

    scheduler.run_next()

    print(scheduler.get_status("bad"))
    # FAILED