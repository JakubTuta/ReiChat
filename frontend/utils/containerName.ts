// Mirrors ContainerManager.container_name on the backend: Docker names allow
// [a-zA-Z0-9_.-] only, and custom models can be namespaced
// ("hf.co/user/repo"). Must stay identical — it's how a model card finds its
// container and its download progress.
export function containerNameFor(model: string, parameters: string): string {
  return `${model}_${parameters}`.replace(/[^\w.-]/g, '-')
}
