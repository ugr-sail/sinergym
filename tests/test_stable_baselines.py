from importlib import metadata

import numpy as np
import pytest

required = {'stable_baselines3'}
installed = {dist.metadata['Name'].lower() for dist in metadata.distributions()}

if required <= installed:
    import stable_baselines3  # type: ignore
    from stable_baselines3.common.noise import NormalActionNoise  # type: ignore

    import sinergym

    TIMESTEPS = 100

    @pytest.mark.parametrize(
        'env_name',
        [
            ('env_demo'),
            ('env_demo_discrete'),
        ],
    )
    def test_stable_PPO(env_name, request):
        env = request.getfixturevalue(env_name)
        model = stable_baselines3.PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

        model.learn(total_timesteps=TIMESTEPS)

        # Check model state
        assert model.action_space == env.action_space
        assert model.env is not None
        assert model.env.action_space == env.action_space

        assert isinstance(
            model.policy, stable_baselines3.common.policies.ActorCriticPolicy
        )  # type: ignore

        # Check model works

        obs, info = env.reset()
        assert info['timestep'] == 0
        a, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(a)

        assert reward is not None and reward < 0
        assert a is not None
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info['timestep'] == 1

        env.close()

    @pytest.mark.parametrize(
        'env_name',
        [
            ('env_demo'),
            ('env_demo_discrete'),
        ],
    )
    def test_stable_A2C(env_name, request):
        env = request.getfixturevalue(env_name)
        model = stable_baselines3.A2C(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            rms_prop_eps=1e-05,
        )

        model.learn(total_timesteps=TIMESTEPS)

        # Check model state
        assert model.action_space == env.action_space
        assert model.env is not None
        assert model.env.action_space == env.action_space

        assert isinstance(
            model.policy, stable_baselines3.common.policies.ActorCriticPolicy
        )  # type: ignore

        # Check model works

        obs, info = env.reset()
        assert info['timestep'] == 0
        a, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(a)

        assert reward is not None and reward < 0
        assert a is not None
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info['timestep'] == 1

        env.close()

    @pytest.mark.parametrize(
        'env_name',
        [
            ('env_demo'),
            ('env_demo_discrete'),
        ],
    )
    def test_stable_DQN(env_name, request):
        env = request.getfixturevalue(env_name)
        # DQN must fail in continuous environments
        if env_name == 'env_demo':
            with pytest.raises(AssertionError):
                model = stable_baselines3.DQN(
                    'MlpPolicy',
                    env,
                    verbose=1,
                    learning_rate=0.0001,
                    buffer_size=1000000,
                    learning_starts=50000,
                    batch_size=32,
                    tau=1.0,
                    gamma=0.99,
                    train_freq=4,
                    gradient_steps=1,
                    target_update_interval=10000,
                    exploration_fraction=0.1,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    max_grad_norm=10,
                )

        else:
            model = stable_baselines3.DQN(
                'MlpPolicy',
                env,
                verbose=1,
                learning_rate=0.0001,
                buffer_size=1000000,
                learning_starts=50000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=10000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10,
            )

            model.learn(total_timesteps=TIMESTEPS)

            # Check model state
            assert model.action_space == env.action_space
            assert model.env is not None
            assert model.env.action_space == env.action_space

            assert isinstance(
                model.policy, stable_baselines3.dqn.policies.DQNPolicy
            )  # type: ignore

            # Check if model works

            obs, info = env.reset()
            assert info['timestep'] == 0
            a, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(a)

            assert reward is not None and reward < 0
            assert a is not None
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert info['timestep'] == 1

            env.close()

    @pytest.mark.parametrize(
        'env_name',
        [
            ('env_demo'),
            ('env_demo_discrete'),
        ],
    )
    def test_stable_DDPG(env_name, request):

        env = request.getfixturevalue(env_name)
        # DDPG must fail in discrete environments
        if env_name == 'env_demo_discrete':
            with pytest.raises(IndexError):
                env.action_space.shape[-1]
            with pytest.raises(AssertionError):
                model = stable_baselines3.DDPG("MlpPolicy", env, verbose=1)
        else:
            # Action noise
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
            # model
            model = stable_baselines3.DDPG(
                "MlpPolicy", env, action_noise=action_noise, verbose=1
            )

            model.learn(total_timesteps=TIMESTEPS)

            # Check model state
            assert model.action_space == env.action_space
            assert model.env is not None
            assert model.env.action_space == env.action_space

            assert isinstance(
                model.policy, stable_baselines3.td3.policies.TD3Policy
            )  # type: ignore

            # Check model works

            obs, info = env.reset()
            assert info['timestep'] == 0
            a, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(a)

            assert reward is not None and reward < 0
            assert a is not None
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert info['timestep'] == 1

            env.close()

    @pytest.mark.parametrize(
        'env_name',
        [
            ('env_demo'),
            ('env_demo_discrete'),
        ],
    )
    def test_stable_SAC(env_name, request):
        env = request.getfixturevalue(env_name)
        # SAC must fail in discrete environments
        if env_name == 'env_demo_discrete':
            with pytest.raises(AssertionError):
                model = stable_baselines3.SAC("MlpPolicy", env, verbose=1)
        else:
            # model
            model = stable_baselines3.SAC("MlpPolicy", env, verbose=1)

            model.learn(total_timesteps=TIMESTEPS)

            # Check model state
            assert model.action_space == env.action_space
            assert model.env is not None
            assert model.env.action_space == env.action_space

            assert isinstance(
                model.policy, stable_baselines3.sac.policies.SACPolicy
            )  # type: ignore

            # Check if model works

            obs, info = env.reset()
            assert info['timestep'] == 0
            a, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(a)

            assert reward is not None and reward < 0
            assert a is not None
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert info['timestep'] == 1

            env.close()
