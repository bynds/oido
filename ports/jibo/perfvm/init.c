/* init.c: copied from bynds/needle-rs (branch claude/port-creation-y2ehfd, commit dcb8f79), ports/jibo/perfvm;
 * only this provenance note is new.
 *
 * /init of the instruction-counting VM (perfvm/run.sh). Runs each line of /jobs (fields
 * separated by tabs) as a command with LD_LIBRARY_PATH set for the armhf sysroot, prints its exit
 * status, and powers the VM off. Static, armhf: the arm64 guest kernel runs it in AArch32 EL0. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/reboot.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

int main(void) {
  static char line[8192];
  char *envp[] = {"LD_LIBRARY_PATH=/lib/arm-linux-gnueabihf:/usr/lib/arm-linux-gnueabihf", "PATH=/bin", "HOME=/",
                  NULL};
  FILE *f;
  mkdir("/proc", 0555);
  mount("proc", "/proc", "proc", 0, NULL);
  setvbuf(stdout, NULL, _IONBF, 0);
  f = fopen("/jobs", "r");
  if (!f) {
    printf("perfvm: no /jobs\n");
  } else {
    while (fgets(line, sizeof line, f)) {
      char *argv[64], *p = line, *tok;
      int argc = 0, st = 0;
      pid_t pid;
      line[strcspn(line, "\n")] = 0;
      if (!line[0] || line[0] == '#') continue;
      while ((tok = strsep(&p, "\t")) && argc < 63) argv[argc++] = tok;
      argv[argc] = NULL;
      printf("perfvm: job %s\n", line);
      pid = fork();
      if (pid == 0) {
        if (chdir("/work") != 0) perror("chdir /work");
        execve(argv[0], argv, envp);
        perror("execve");
        _exit(127);
      }
      waitpid(pid, &st, 0);
      printf("perfvm: exit %d\n", WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st));
    }
    fclose(f);
  }
  sync();
  reboot(RB_POWER_OFF);
  return 0;
}
