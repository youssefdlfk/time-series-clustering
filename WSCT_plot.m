function WSCT_plot(MDP, f_act, f_state, mod_out, timestep_to_plot)
% f_act: state factor for which to display sampled actions
% mod_out : outcome modality for which to display outcomes

spm_figure('GetWin'); clf    % display behavior


if iscell(MDP(1).X)
    Nf = numel(MDP(1).B);                 % number of hidden state factors
    Ng = numel(MDP(1).A);                 % number of outcome factors
else
    Nf = 1;
    Ng = 1;
end


Nt    = length(MDP);               % number of trials
Ne    = size(MDP(1).V,1) + 1;      % number of epochs per trial
Np    = size(MDP(1).V,2) + 1;      % number of policies


for i = 1:Nt
    % assemble expectations of hidden states and outcomes
    %----------------------------------------------------------------------
    for j = 1:Ne
        for k = 1:Ne
            for f = 1:Nf
                try
                    x{f}{i,1}{k,j} = gradient(MDP(i).xn{f}(:,:,j,k)')';
                catch
                    x{f}{i,1}{k,j} = gradient(MDP(i).xn(:,:,j,k)')';
                end
            end
        end
    end
    act_prob(:,i) = MDP(i).P(:, :, :, :, :, :, timestep_to_plot);
    act(:,i) = MDP(i).u(f_act,timestep_to_plot);
    pref(:, i) = MDP(i).C{mod_out}(:, timestep_to_plot);
    outcomes(:, i) = MDP(i).o(mod_out, timestep_to_plot+1);
    posterior_states(:, i) = MDP(i).X{f_state}(:, timestep_to_plot+1);
    states(:, i) = MDP(i).s(f_state, timestep_to_plot+1);
    dn(:,i) = mean(MDP(i).dn,2);
    wn(:,i) = MDP(i).wn;

end


% Actions
%--------------------------------------------------------------------------
col   = {'r.','g.','b.','c.','m.','k.'};
subplot(6,1,1)
if Nt < 64
    MarkerSize = 24;
else
    MarkerSize = 16;
end

image(64*(1 - act_prob)),  hold on

plot(act,col{1},'MarkerSize',MarkerSize)

try
    plot(Np*(1 - act_prob(Np,:)),'r')
end
try
    E = spm_softmax(spm_cat({MDP.e}));
    plot(Np*(1 - E(end,:)),'r:')
end
title('Action selection and action probabilities')
xlabel('Trial'),ylabel('Action'), hold off
yticks(1:numel(MDP(1).label.action{f_act}))
yticklabels(MDP(1).label.action{f_act})

%============================================

% Outcomes and preference
%--------------------------------------------------------------------------
subplot(6,1,2)
if Nt < 64
    MarkerSize = 24;
else
    MarkerSize = 16;
end

image(64*(1 - pref)),  hold on

plot(outcomes,col{1},'MarkerSize',MarkerSize)


title('Outcomes and outcome probabilities')
xlabel('Trial'),ylabel(MDP(1).label.modality{mod_out}), hold off
yticks(1:numel(MDP(1).label.modality{mod_out}))
yticklabels(MDP(1).label.outcome{mod_out})


% posterior beliefs about hidden states
%--------------------------------------------------------------------------
subplot(6,1,3)
if Nt < 64
    MarkerSize = 24;
else
    MarkerSize = 16;
end

image(64*(1 - posterior_states)),  hold on

plot(states,col{1},'MarkerSize',MarkerSize)
title('Hidden states and posterior beliefs')
xlabel('Trial'),ylabel(MDP(1).label.factor{f_state}), hold off
yticks(1:numel(MDP(1).label.name{f_state}))
yticklabels(MDP(1).label.name{f_state})


% precision
%--------------------------------------------------------------------------
subplot(6,1,4)
w = dn;
w   = spm_vec(w);
if Nt > 8
    fill([1 1:length(w) length(w)],[0; w.*(w > 0); 0],'k'), hold on
    fill([1 1:length(w) length(w)],[0; w.*(w < 0); 0],'k'), hold off
else
    bar(w,1.1,'k')
end
title('Precision (dopamine)')
ylabel('Precision'), spm_axis tight, box off
YLim = get(gca,'YLim'); YLim(1) = 0; set(gca,'YLim',YLim);
set(gca,'XTickLabel',{});


% free energy & confidence
% ------------------------------------------------------------------
[F,Fu,~,~,~,~] = spm_MDP_F(MDP);
subplot(6,1,5), plot(1:Nt,F),  xlabel('trial'), spm_axis tight, title('Free energy')
subplot(6,1,6), plot(1:Nt,Fu), xlabel('trial'), spm_axis tight, title('Confidence')
% subplot(5,1,5), plot(1:Nt,Fa(mod_out:mod_out:end)), xlabel('trial'), spm_axis tight, title('Free energy of A{'+string(mod_out)+'} parameters')


% spm_figure('Matrices','WSCT'); clf    % display behavior

% Learned likelihood matrix a
% --------------------------------------------------------------------
% trial_to_plot = 4;
% a = MDP(trial_to_plot).a;
% subplot(2, 1, 1)
% for i = 1:4                                                                                                                        
%     for j= 1:8
%         a{i,j} = squeeze(a{4}(:,i,j,3));
%         a{i,j} = a{i,j}*diag(1./sum(a{i,j}));
%     end
% end
% a = spm_cat(a);
% imagesc(a);
% title( 'Sample: left - center - right', 'FontSize',16)
% ylabel('Rule: left - center - right','FontSize',14)
% xlabel('Correct color', 'FontSize',14)
% set(gca,'XTick',1:9)
% set(gca,'YTick',1:12)
% set(gca,'XTicklabel',repmat(['r','g','b'],[1 3])')
% set(gca,'YTicklabel',repmat(['r','g','b',' '],[1 3])')
% axis image


% Learned likelihood matrix
% --------------------------------------------------------------------
% subplot(8,1,8)
% Z = MDP(Nt).a{1}(:, 1, 1, 1, 2, 2, 2, 1, 2, 2, :);
% image(64*(1 - Z))
% title('Likelihood matrix')
% ylabel('feedback')
% yticks([1, 2, 3])
% yticklabels({'incorrect', 'correct', 'null'})
% xlabel('choice state')


% changes in expected precision
% ------------------------------------------------------------------
% subplot(3,1,2)
% dn    = spm_vec(dn);
% dn    = dn.*(dn > 0);
% dn    = dn + (dn + 1/16).*rand(size(dn))/8;
% bar(dn,1,'k'), title('Dopamine responses')
% xlabel('time (updates)','FontSize',12)
% ylabel('change in precision','FontSize',12), spm_axis tight, box off
% YLim = get(gca,'YLim'); YLim(1) = 0; set(gca,'YLim',YLim);

% % posterior beliefs about hidden states
% figure;
% for f = 1:Nf
%     subplot(Nf, 1, f)
%     image(64*(1 - X{gf(f)})), hold on
%     if size(X{gf(f)},1) > 128
%         spm_spy(X{gf(f)},16,1)
%     end
%     plot(MDP.s(gf(f),:),'.c','MarkerSize',16), hold off
%     if f < 2
%         title(sprintf('Hidden states - %s',MDP.label.factor{gf(f)}));
%     else
%         title(MDP.label.factor{gf(f)});
%     end
% end
% 
% % outcome and preferences
% figure;
% for g = 1:Ng
%     subplot(Ng, 1, g)
%     image(64*(1 - C{gg(g)})), hold on
%     if size(C{gg(g)},1) > 128
%         spm_spy(C{gg(g)},16,1)
%     end
%     plot(MDP.o(gg(g),:),'.c','MarkerSize',16), hold off
%     if f < 2
%         title(sprintf('Hidden states - %s',MDP.label.modality{gg(g)}));
%     else
%         title(MDP.label.modality{gg(g)});
%     end
% end
% 
% 
% % posterior beliefs about control states
% %--------------------------------------------------------------------------
% figure;
% Nu     = find(Nu);
% Np     = length(Nu);
% for f  = 1:Np
%     subplot(Np,1,f)
%     P  = MDP.P;
%     if Nf > 1
%         ind     = 1:Nf;
%         for dim = 1:Nf
%             if dim ~= ind(Nu(f))
%                 P = sum(P,dim);
%             end
%         end
%         P = squeeze(P);
%     end
% 
%     % display
%     %----------------------------------------------------------------------
%     image(64*(1 - P)), hold on
%     plot(MDP.u(Nu(f),:),'.c','MarkerSize',16), hold off
%     if f < 2
%         title(sprintf('Action - %s',MDP.label.factor{Nu(f)}));
%     else
%         title(MDP.label.factor{Nu(f)});
%     end
%     set(gca,'XTickLabel',{});
%     set(gca,'XTick',1:size(X{1},2));
%     set(gca,'YTick',1:numel(MDP.label.action{Nu(f)}));
%     set(gca,'YTickLabel',MDP.label.action{Nu(f)});
% 
% 
% end