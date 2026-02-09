#pragma once
#include <pthread.h>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include "core/custom_concepts.hpp"

namespace pthreads_manage {

    static std::size_t get_count_cpu() {
        const std::size_t count_cpu = sysconf(_SC_NPROCESSORS_ONLN);
        return count_cpu;
    }

    struct PartitionerSettings {
        std::size_t full_size_;
        std::size_t chunk_size_;
        std::size_t overlap_size_;
    };

    struct JobContext {
        ViewType parent_view; //Откуда нарезать сегменты

        void (*run_kernel)(ViewType chunk, std::size_t worker_id, void* kernel_args); //Прокладка, запускающая ядро
        void* kernel_args;

        PartitionerSettings (*partitioner)(void* partitioner_args); //Разделитель, возвращающий параметры для нарезки сегментов
        void* partitioner_args;

    };

    class Pool {
    public:
        [[nodiscard]] std::size_t totalThreads() const {return total_count_threads_;}
        ///Постановка задачи с main thread
        void dispatchJob(const JobContext& job) noexcept {
            pthread_mutex_lock(&mutex_); // для защиты job, job_active, active_workers + mutex необходим для условия

            current_job_ = job;
            job_active_ = true;
            active_workers_ = 1;
            job_id_++;
            settings_ = job.partitioner(job.partitioner_args);

            //main thread тоже запускается. чтобы не ждал вхолостую
            std::size_t begin_subrange = 0;
            std::size_t end_subrange = std::min(begin_subrange + settings_.chunk_size_, job.parent_view.extent(0));
            auto subrange = Kokkos::subview(
                                                    job.parent_view,
                                                        Kokkos::pair(begin_subrange, end_subrange),
                                                        Kokkos::ALL
                                                    );
            pthread_cond_broadcast(&job_start_);
            pthread_mutex_unlock(&mutex_); // Открываем мутекс, чтобы остальные не ждали пока main выполнит ядро
            job.run_kernel(subrange, 0, job.kernel_args);

            pthread_mutex_lock(&mutex_); // Закрываем для проверки условия
            if (--active_workers_ == 0) {
                job_active_ = false;
                pthread_cond_signal(&job_done_);
            }
            while(job_active_)
                pthread_cond_wait(&job_done_, &mutex_);
                //для условия мутекс нужен, поскольку условие ожидания может изменится в момент проверки = вечный сон ожидающего потока
            pthread_mutex_unlock(&mutex_);
        }

        explicit Pool() noexcept : contexts_(get_count_cpu()), total_count_threads_(get_count_cpu()), threads_(get_count_cpu()) {
            for (std::size_t tid = 1; tid < total_count_threads_ - 1; ++tid) {
                pthread_attr_t attr;
                pthread_attr_init(&attr);
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(tid, &set);
                pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
                contexts_[tid] = WorkerContext{this, tid};

                //Создаем поток и забрасываем его в пул this
                pthread_create(&threads_[tid], &attr, &Pool::workerEntry, &contexts_[tid]);
                pthread_attr_destroy(&attr);
            }
        }
        ~Pool() noexcept {
            pthread_mutex_lock(&mutex_);
            stop_ = true;
            job_active_ = false;
            pthread_cond_broadcast(&job_start_);
            pthread_mutex_unlock(&mutex_);

            for (auto& t : threads_) {
                pthread_join(t, nullptr);
            }

            pthread_mutex_destroy(&mutex_);
            pthread_cond_destroy(&job_start_);
            pthread_cond_destroy(&job_done_);
        }

    private:
        ///Прокладка между бесконечным циклом ожидания задачи и созданием потока, чтобы сделать цикл членом класса
        static void* workerEntry(void* arg) noexcept { // static поскольку pthread_create требует указатель на функцию (без static тип: void* (PthreadPool::*)(void*))
            auto* context_ptr = static_cast<WorkerContext*>(arg);
            context_ptr->pool_->workerLoop(context_ptr->tid_);
            return nullptr;
        }
        ///Бесконечный цикл ожидания задачи
        void workerLoop(std::size_t worker_id) noexcept {
            std::size_t local_job_id = 0;
            for (;;) {
                //Невозможно проверить условие без мутекса
                pthread_mutex_lock(&mutex_);
                // Нет задачи и еще не закончили => ждем
                //Чтобы потоки не выходили и заходили несколько раз в одну и ту же задачу
                while (local_job_id == job_id_ && !stop_)
                    pthread_cond_wait(&job_start_, &mutex_);
                // Закончили
                if (stop_) {
                    pthread_mutex_unlock(&mutex_);
                    break;
                }
                //Если дошли до этой точки, значит появилась задача (разбудил главный поток)
                auto job = current_job_;
                ++active_workers_;
                local_job_id = job_id_; //Чтобы потоки не выходили и заходили несколько раз в одну и ту же задачу

                std::size_t begin_subrange = worker_id * settings_.chunk_size_ - settings_.overlap_size_ * worker_id;
                std::size_t end_subrange = std::min(begin_subrange + settings_.chunk_size_, job.parent_view.extent(0));
                auto subrange = Kokkos::subview(
                                                        job.parent_view,
                                                            Kokkos::pair(begin_subrange, end_subrange),
                                                            Kokkos::ALL
                                                        );
                pthread_mutex_unlock(&mutex_);

                job.run_kernel(subrange, worker_id, job.kernel_args);

                //Окончание работы
                pthread_mutex_lock(&mutex_);
                if (--active_workers_ == 0) {
                    job_active_ = false;
                    pthread_cond_signal(&job_done_); //Последний вышедший сигнализирует в main
                }
                pthread_mutex_unlock(&mutex_);
            }
        }

        struct WorkerContext {
            Pool* pool_;
            std::size_t tid_;
        };
        std::vector<WorkerContext> contexts_; // Для передачи tid

        ///Достпуные потоки в системе
        const std::size_t total_count_threads_;
        std::vector<pthread_t> threads_;

        JobContext current_job_{};
        std::size_t job_id_{}; // чтобы отличать текущую задачу от предыдущей

        pthread_mutex_t mutex_ = PTHREAD_MUTEX_INITIALIZER;
        pthread_cond_t job_start_ = PTHREAD_COND_INITIALIZER;
        pthread_cond_t job_done_ = PTHREAD_COND_INITIALIZER;

        volatile bool job_active_{false};
        volatile bool stop_{false};

        PartitionerSettings settings_{};
        std::size_t active_workers_{0};

    };

}